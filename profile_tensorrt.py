"""Profile the Wan 2.2 VAE with torch_tensorrt as the torch.compile backend.

Same structure as profile_diffusers_compile.py but uses TensorRT-optimized
kernels via torch_tensorrt. Requires:
    pip install torch_tensorrt

The Wan VAE's mutable `feat_cache` list forces graph breaks, so TensorRT
optimizes the compute subgraphs (conv3d, attention, residual blocks) between
breaks rather than the full encoder/decoder. Even with this limitation, TRT's
kernel library (fused conv+activation, TRT's own attention kernels) is what we
want to benchmark against stock inductor.
"""

import argparse
import os
import sys
import time

import torch
import torch._dynamo
from diffusers import AutoencoderKLWan

# Same reasoning as in profile_diffusers_compile.py: the VAE forces many
# per-shape recompiles of the shared residual/conv forwards.
torch._dynamo.config.recompile_limit = 64
torch._dynamo.config.accumulated_recompile_limit = 1024

try:
    import torch_tensorrt  # noqa: F401
except ImportError:
    sys.stderr.write(
        "torch_tensorrt is not installed.\n"
        "Install it with:\n"
        "    pip install torch_tensorrt --extra-index-url "
        "https://download.pytorch.org/whl/cu128\n"
    )
    sys.exit(2)

DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def nvtx_push(name: str) -> None:
    torch.cuda.nvtx.range_push(name)


def nvtx_pop() -> None:
    torch.cuda.nvtx.range_pop()


def cuda_profiler_start() -> None:
    torch.cuda.cudart().cudaProfilerStart()


def cuda_profiler_stop() -> None:
    torch.cuda.cudart().cudaProfilerStop()


def instrumented_encode(vae: AutoencoderKLWan, x: torch.Tensor) -> torch.Tensor:
    _, _, num_frame, _, _ = x.shape
    vae.clear_cache()

    iter_ = 1 + (num_frame - 1) // 4
    out = None
    for i in range(iter_):
        vae._enc_conv_idx = [0]
        nvtx_push(f"enc_chunk_{i}")
        if i == 0:
            out = vae.encoder(
                x[:, :, :1, :, :],
                feat_cache=vae._enc_feat_map,
                feat_idx=vae._enc_conv_idx,
            )
        else:
            out_ = vae.encoder(
                x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                feat_cache=vae._enc_feat_map,
                feat_idx=vae._enc_conv_idx,
            )
            out = torch.cat([out, out_], 2)
        nvtx_pop()

    nvtx_push("enc_quant_conv")
    enc = vae.quant_conv(out)
    nvtx_pop()
    vae.clear_cache()
    return enc


def instrumented_decode(vae: AutoencoderKLWan, z: torch.Tensor) -> torch.Tensor:
    _, _, num_frame, _, _ = z.shape
    vae.clear_cache()

    nvtx_push("dec_post_quant_conv")
    x = vae.post_quant_conv(z)
    nvtx_pop()

    out = None
    for i in range(num_frame):
        vae._conv_idx = [0]
        nvtx_push(f"dec_chunk_{i}")
        if i == 0:
            out = vae.decoder(
                x[:, :, i : i + 1, :, :],
                feat_cache=vae._feat_map,
                feat_idx=vae._conv_idx,
                first_chunk=True,
            )
        else:
            out_ = vae.decoder(
                x[:, :, i : i + 1, :, :],
                feat_cache=vae._feat_map,
                feat_idx=vae._conv_idx,
            )
            out = torch.cat([out, out_], 2)
        nvtx_pop()

    out = torch.clamp(out, min=-1.0, max=1.0)
    vae.clear_cache()
    return out


def apply_channels_last_3d(model: torch.nn.Module) -> torch.nn.Module:
    """Convert only the 5D parameters/buffers to channels_last_3d."""
    for p in model.parameters():
        if p.dim() == 5:
            p.data = p.data.to(memory_format=torch.channels_last_3d)
    for b in model.buffers():
        if b.dim() == 5:
            b.data = b.data.to(memory_format=torch.channels_last_3d)
    return model


def time_pass(fn, *args) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn(*args)
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="bf16")
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--skip-encode", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--channels-last", action="store_true",
                        help="Apply channels_last_3d to 5D VAE weights/inputs. "
                             "Off by default.")
    parser.add_argument("--min-block-size", type=int, default=5,
                        help="torch_tensorrt min_block_size option")
    args = parser.parse_args()

    assert (args.num_frames - 1) % 4 == 0
    assert args.height % 8 == 0 and args.width % 8 == 0

    device = torch.device("cuda")
    dtype = DTYPE_MAP[args.dtype]
    torch.set_grad_enabled(False)

    trt_precisions = {dtype}
    if dtype != torch.float32:
        trt_precisions.add(torch.float32)  # allow fallback for ops TRT can't run in fp16

    print(f"[config] backend=torch_tensorrt dtype={dtype} "
          f"shape=[1,3,{args.num_frames},{args.height},{args.width}]")

    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=dtype
    ).to(device).eval()
    if args.channels_last:
        apply_channels_last_3d(vae)
        print("[config] applied channels_last_3d to 5D VAE weights")
    print(f"[load] VAE loaded in {time.time() - t0:.2f}s")

    print(f"[compile] compiling encoder/decoder with torch_tensorrt "
          f"(min_block_size={args.min_block_size})...")
    print("[compile] Note: first encode/decode pass will trigger TRT engine "
          "builds per subgraph. Expect several minutes of warmup at full "
          "720p/77-frame resolution.")
    compile_opts = {
        "enabled_precisions": trt_precisions,
        "min_block_size": args.min_block_size,
        "truncate_long_and_double": True,
        "use_fp32_acc": False,
    }
    # torch.compile with backend="torch_tensorrt" triggers lowering via
    # torch_tensorrt.dynamo. Graph breaks caused by the mutable feat_cache
    # list result in multiple TRT-optimized subgraphs per call.
    vae.encoder = torch.compile(
        vae.encoder, backend="torch_tensorrt",
        options=compile_opts, dynamic=False,
    )
    vae.decoder = torch.compile(
        vae.decoder, backend="torch_tensorrt",
        options=compile_opts, dynamic=False,
    )

    latent_t = 1 + (args.num_frames - 1) // 4
    latent_h = args.height // 8
    latent_w = args.width // 8
    print(f"[shapes] pixel=[1,3,{args.num_frames},{args.height},{args.width}] "
          f"latent=[1,16,{latent_t},{latent_h},{latent_w}]")

    def make_pixel() -> torch.Tensor:
        t = torch.randn(
            (1, 3, args.num_frames, args.height, args.width),
            device=device, dtype=dtype,
        )
        if args.channels_last:
            t = t.to(memory_format=torch.channels_last_3d)
        return t

    def make_latent() -> torch.Tensor:
        t = torch.randn(
            (1, 16, latent_t, latent_h, latent_w),
            device=device, dtype=dtype,
        )
        if args.channels_last:
            t = t.to(memory_format=torch.channels_last_3d)
        return t

    # --- Encode --------------------------------------------------------------
    if not args.skip_encode:
        print("\n=== ENCODE (prefill) ===")
        x = make_pixel()
        for w in range(args.warmup):
            t0 = time.time()
            _, ms = time_pass(instrumented_encode, vae, x)
            print(f"[encode warmup {w}] {ms:.2f} ms (wall {time.time() - t0:.2f}s)")

        torch.cuda.reset_peak_memory_stats()
        cuda_profiler_start()
        nvtx_push("encode_measure")
        encode_times = []
        for i in range(args.repeat):
            nvtx_push(f"encode_iter_{i}")
            _, ms = time_pass(instrumented_encode, vae, x)
            nvtx_pop()
            encode_times.append(ms)
            print(f"[encode iter {i}] {ms:.2f} ms")
        nvtx_pop()
        cuda_profiler_stop()

        mean = sum(encode_times) / len(encode_times)
        print(f"[encode] mean={mean:.2f} ms min={min(encode_times):.2f} "
              f"max={max(encode_times):.2f}")
        print(f"[encode] peak_alloc={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        del x

    # --- Decode --------------------------------------------------------------
    if not args.skip_decode:
        print("\n=== DECODE ===")
        z = make_latent()
        for w in range(args.warmup):
            t0 = time.time()
            _, ms = time_pass(instrumented_decode, vae, z)
            print(f"[decode warmup {w}] {ms:.2f} ms (wall {time.time() - t0:.2f}s)")

        torch.cuda.reset_peak_memory_stats()
        cuda_profiler_start()
        nvtx_push("decode_measure")
        decode_times = []
        for i in range(args.repeat):
            nvtx_push(f"decode_iter_{i}")
            _, ms = time_pass(instrumented_decode, vae, z)
            nvtx_pop()
            decode_times.append(ms)
            print(f"[decode iter {i}] {ms:.2f} ms")
        nvtx_pop()
        cuda_profiler_stop()

        mean = sum(decode_times) / len(decode_times)
        print(f"[decode] mean={mean:.2f} ms min={min(decode_times):.2f} "
              f"max={max(decode_times):.2f}")
        print(f"[decode] peak_alloc={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
