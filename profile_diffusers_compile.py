"""Profile the Wan 2.2 VAE from diffusers with channels_last_3d + torch.compile.

Profiles encode (prefill) and decode separately using NVTX markers so that
Nsight Systems / Nsight Compute can be used to inspect the breakdown.

Only the VAE is profiled — no text encoder, no DiT, no scheduler.
"""

import argparse
import ctypes
import os
import time

import torch
import torch._dynamo
from diffusers import AutoencoderKLWan

# The VAE reuses WanResidualBlock / WanCausalConv3d.forward across blocks with
# different channel counts (96, 192, 384, 768, ...). Each variant triggers a
# recompile; the default limit of 8 is hit mid-encoder and dynamo falls back
# to eager for later blocks. Lift both limits so all blocks get compiled.
torch._dynamo.config.recompile_limit = 64
torch._dynamo.config.accumulated_recompile_limit = 1024

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
    """Reimplementation of `AutoencoderKLWan._encode` with per-chunk NVTX ranges.

    Matches the upstream logic in diffusers 0.37 (autoencoder_kl_wan.py:1128-1153).
    The causal encoder is called once for the first frame and once for every
    4-frame chunk after it; the feature cache is mutated in place between calls.
    """
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
    """Reimplementation of `AutoencoderKLWan._decode` with per-chunk NVTX ranges.

    Matches the upstream logic in diffusers 0.37 (autoencoder_kl_wan.py:1182-1211).
    """
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
    """Convert only the 5D parameters/buffers to channels_last_3d.

    `model.to(memory_format=torch.channels_last_3d)` errors on recent PyTorch
    when any parameter is not rank 5, which is the case here (the VAE has
    Conv2d inside attention blocks).
    """
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
    parser.add_argument("--num-frames", type=int, default=81,
                        help="Must satisfy (N-1) % 4 == 0 (e.g. 1, 5, 9, ..., 77).")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip torch.compile (baseline run).")
    parser.add_argument("--no-channels-last", action="store_true",
                        help="Skip channels_last_3d memory format.")
    parser.add_argument("--compile-mode", default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="'reduce-overhead' enables CUDA graphs but conflicts "
                             "with the VAE's in-place feature cache mutation.")
    parser.add_argument("--skip-encode", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()

    assert (args.num_frames - 1) % 4 == 0, \
        f"num_frames={args.num_frames} must satisfy (N-1) % 4 == 0"
    assert args.height % 8 == 0 and args.width % 8 == 0, "H/W must be multiples of 8"

    device = torch.device("cuda")
    dtype = DTYPE_MAP[args.dtype]
    torch.set_grad_enabled(False)

    print(f"[config] model={args.model_id} dtype={dtype} "
          f"shape=[1,3,{args.num_frames},{args.height},{args.width}] "
          f"compile={not args.no_compile} channels_last={not args.no_channels_last}")

    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=dtype
    ).to(device).eval()
    print(f"[load] VAE loaded in {time.time() - t0:.2f}s")

    if not args.no_channels_last:
        apply_channels_last_3d(vae)
        print("[config] applied channels_last_3d to 5D VAE weights")

    if not args.no_compile:
        print(f"[compile] compiling encoder and decoder (mode={args.compile_mode})...")
        # The outer _encode/_decode loops manage mutable cache state; compile
        # the inner encoder/decoder submodules which do the heavy compute.
        vae.encoder = torch.compile(
            vae.encoder, backend="inductor",
            mode=args.compile_mode, fullgraph=False,
        )
        vae.decoder = torch.compile(
            vae.decoder, backend="inductor",
            mode=args.compile_mode, fullgraph=False,
        )

    # Latent spatial dims follow the VAE's 8x / 4x compression ratios.
    latent_t = 1 + (args.num_frames - 1) // 4
    latent_h = args.height // 8
    latent_w = args.width // 8
    print(f"[shapes] pixel=[1,3,{args.num_frames},{args.height},{args.width}] "
          f"latent=[1,16,{latent_t},{latent_h},{latent_w}]")

    pixel_shape = (1, 3, args.num_frames, args.height, args.width)
    latent_shape = (1, 16, latent_t, latent_h, latent_w)

    def make_pixel() -> torch.Tensor:
        t = torch.randn(pixel_shape, device=device, dtype=dtype)
        if not args.no_channels_last:
            t = t.to(memory_format=torch.channels_last_3d)
        return t

    def make_latent() -> torch.Tensor:
        t = torch.randn(latent_shape, device=device, dtype=dtype)
        if not args.no_channels_last:
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
            latent, ms = time_pass(instrumented_encode, vae, x)
            nvtx_pop()
            encode_times.append(ms)
            print(f"[encode iter {i}] {ms:.2f} ms")
        nvtx_pop()
        cuda_profiler_stop()

        mean = sum(encode_times) / len(encode_times)
        print(f"[encode] mean={mean:.2f} ms min={min(encode_times):.2f} "
              f"max={max(encode_times):.2f}")
        print(f"[encode] peak_alloc={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB "
              f"peak_reserved={torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")
        del x, latent

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
            decoded, ms = time_pass(instrumented_decode, vae, z)
            nvtx_pop()
            decode_times.append(ms)
            print(f"[decode iter {i}] {ms:.2f} ms")
        nvtx_pop()
        cuda_profiler_stop()

        mean = sum(decode_times) / len(decode_times)
        print(f"[decode] mean={mean:.2f} ms min={min(decode_times):.2f} "
              f"max={max(decode_times):.2f}")
        print(f"[decode] peak_alloc={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB "
              f"peak_reserved={torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
