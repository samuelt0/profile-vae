"""Profile the Wan 2.2 VAE using SGLang's standalone implementation.

SGLang (sglang.multimodal_gen) ships its own AutoencoderKLWan in
    .../sglang/multimodal_gen/runtime/models/vaes/wanvae.py

It uses contextvars for feature caching rather than explicit arguments, so we
reimplement the chunked encode/decode loops inside `forward_context(...)`.

Weights are loaded from the diffusers HF cache; the state dict layout is the
same between diffusers and SGLang implementations.
"""

import argparse
import glob
import os
import sys
import time
import types
from pathlib import Path

import torch
import torch._dynamo
from safetensors.torch import load_file

# Lift per-function recompile cap: WanResidualBlock / WanCausalConv3d are
# reused at several channel counts (96, 192, 384, 768). Default 8 is hit
# mid-encoder and dynamo falls back to eager for later blocks.
torch._dynamo.config.recompile_limit = 64
torch._dynamo.config.accumulated_recompile_limit = 1024


def _install_sglang_stubs() -> None:
    """Install stubs so we can import the SGLang VAE without pulling in the
    full server stack.

    Why this is needed: the SGLang install in this conda env has several
    broken downstream deps (sgl_kernel built for an older libtorch, transformers
    missing GptOssConfig, etc.). None of that is used by the VAE, but the
    transitive imports fail anyway. We stub the problematic modules with the
    minimum surface the VAE actually touches.
    """
    import sglang as _sglang

    mm_path = os.path.join(os.path.dirname(_sglang.__file__), "multimodal_gen")
    runtime_path = os.path.join(mm_path, "runtime")

    # sgl_kernel: VAE doesn't actually call silu_and_mul (it uses nn.SiLU via
    # get_act_fn), but activation.py imports it at module scope.
    sgl_kernel_stub = types.ModuleType("sgl_kernel")
    sgl_kernel_stub.silu_and_mul = lambda *a, **k: None
    sys.modules.setdefault("sgl_kernel", sgl_kernel_stub)

    # sglang.multimodal_gen: skip the eager DiffGenerator/server import chain.
    mm_stub = types.ModuleType("sglang.multimodal_gen")
    mm_stub.__path__ = [mm_path]
    sys.modules.setdefault("sglang.multimodal_gen", mm_stub)

    # sglang.multimodal_gen.runtime: there is no __init__.py, but we still
    # need it registered so the subpackage lookup resolves without running
    # platforms/__init__.py (which pulls in pynvml plugins).
    runtime_stub = types.ModuleType("sglang.multimodal_gen.runtime")
    runtime_stub.__path__ = [runtime_path]
    sys.modules.setdefault("sglang.multimodal_gen.runtime", runtime_stub)

    # Stub runtime.platforms with minimal enums and current_platform. wanvae.py
    # imports current_platform but the VAE code never actually branches on it
    # for the cases we care about. DiTConfig imports AttentionBackendEnum for
    # annotations only.
    import enum as _enum
    platforms_stub = types.ModuleType("sglang.multimodal_gen.runtime.platforms")

    class _AttentionBackendEnum(_enum.Enum):
        FLASH_ATTN = "flash_attn"
        TORCH_SDPA = "torch_sdpa"

    class _PlatformEnum(_enum.Enum):
        CUDA = "cuda"
        ROCM = "rocm"
        CPU = "cpu"
        UNSPECIFIED = "unspecified"

    class _Platform:
        _enum = _PlatformEnum.CUDA
        def is_cuda(self) -> bool:
            return True
        def is_rocm(self) -> bool:
            return False
        def is_cpu(self) -> bool:
            return False
        def is_mps(self) -> bool:
            return False
        def is_tpu(self) -> bool:
            return False
        def is_cuda_alike(self) -> bool:
            return True

    platforms_stub.AttentionBackendEnum = _AttentionBackendEnum
    platforms_stub.PlatformEnum = _PlatformEnum
    platforms_stub.Platform = _Platform
    platforms_stub.current_platform = _Platform()
    sys.modules.setdefault(
        "sglang.multimodal_gen.runtime.platforms", platforms_stub
    )

    # Stub runtime.distributed with trivial SP rank/size. The VAE base class
    # calls get_sp_world_size() to decide whether to use parallel tiling;
    # returning 1 forces the non-parallel path, which is what we want anyway.
    distributed_stub = types.ModuleType(
        "sglang.multimodal_gen.runtime.distributed"
    )
    distributed_stub.get_sp_world_size = lambda: 1
    distributed_stub.get_sp_parallel_rank = lambda: 0
    sys.modules.setdefault(
        "sglang.multimodal_gen.runtime.distributed", distributed_stub
    )


_install_sglang_stubs()

from sglang.multimodal_gen.configs.models.vaes.wanvae import (  # noqa: E402
    WanVAEArchConfig,
    WanVAEConfig,
)
from sglang.multimodal_gen.runtime.models.vaes.wanvae import (  # noqa: E402
    AutoencoderKLWan,
    feat_idx,
    first_chunk,
    forward_context,
)

DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

HF_CACHE_GLOB = (
    "~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-I2V-A14B-Diffusers/"
    "snapshots/*/vae/diffusion_pytorch_model.safetensors"
)


def locate_vae_weights() -> str:
    matches = glob.glob(os.path.expanduser(HF_CACHE_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"No VAE safetensors found matching {HF_CACHE_GLOB}. "
            "Download the model first with "
            "`huggingface-cli download Wan-AI/Wan2.2-I2V-A14B-Diffusers`."
        )
    return matches[0]


def nvtx_push(name: str) -> None:
    torch.cuda.nvtx.range_push(name)


def nvtx_pop() -> None:
    torch.cuda.nvtx.range_pop()


def cuda_profiler_start() -> None:
    torch.cuda.cudart().cudaProfilerStart()


def cuda_profiler_stop() -> None:
    torch.cuda.cudart().cudaProfilerStop()


def build_vae(dtype: torch.dtype, device: torch.device) -> AutoencoderKLWan:
    # Defaults of WanVAEArchConfig already match Wan 2.2 (base_dim=96, z_dim=16,
    # dim_mult=(1,2,4,4), temperal_downsample=(False,True,True), etc.).
    arch = WanVAEArchConfig()
    config = WanVAEConfig(
        arch_config=arch,
        load_encoder=True,
        load_decoder=True,
        use_feature_cache=True,
        use_tiling=False,
        use_temporal_tiling=False,
        use_parallel_tiling=False,
    )

    with torch.device("meta"):
        vae = AutoencoderKLWan(config)

    weights_path = locate_vae_weights()
    print(f"[load] loading weights from {weights_path}")
    state_dict = load_file(weights_path)

    # Move to real device and load weights (state_dict keys match diffusers).
    vae = vae.to_empty(device=device)
    missing, unexpected = vae.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[load] unexpected keys (first 5): {unexpected[:5]}")
    if missing:
        print(f"[load] missing keys (first 5): {missing[:5]}")
    return vae.to(dtype=dtype).eval()


def instrumented_encode(vae: AutoencoderKLWan, x: torch.Tensor) -> torch.Tensor:
    """SGLang encode with per-chunk NVTX ranges.

    Mirrors AutoencoderKLWan.encode (wanvae.py:1205) for the `use_feature_cache`
    path, with NVTX markers around each chunk iteration.
    """
    vae.clear_cache()

    iter_ = 1 + (x.shape[2] - 1) // 4
    with forward_context(
        feat_cache_arg=vae._enc_feat_map, feat_idx_arg=vae._enc_conv_idx
    ):
        out = None
        for i in range(iter_):
            feat_idx.set(0)
            nvtx_push(f"enc_chunk_{i}")
            if i == 0:
                out = vae.encoder(x[:, :, :1, :, :])
            else:
                out_ = vae.encoder(x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :])
                out = torch.cat([out, out_], 2)
            nvtx_pop()

    nvtx_push("enc_quant_conv")
    enc = vae.quant_conv(out)
    nvtx_pop()
    vae.clear_cache()
    return enc


def instrumented_decode(vae: AutoencoderKLWan, z: torch.Tensor) -> torch.Tensor:
    """SGLang decode with per-chunk NVTX ranges.

    Mirrors AutoencoderKLWan.decode (wanvae.py:1263).
    """
    vae.clear_cache()
    iter_ = z.shape[2]

    nvtx_push("dec_post_quant_conv")
    x = vae.post_quant_conv(z)
    nvtx_pop()

    with forward_context(feat_cache_arg=vae._feat_map, feat_idx_arg=vae._conv_idx):
        out = None
        for i in range(iter_):
            feat_idx.set(0)
            nvtx_push(f"dec_chunk_{i}")
            if i == 0:
                first_chunk.set(True)
                out = vae.decoder(x[:, :, i : i + 1, :, :])
            else:
                first_chunk.set(False)
                out_ = vae.decoder(x[:, :, i : i + 1, :, :])
                out = torch.cat([out, out_], 2)
            nvtx_pop()

    out = out.float().clamp(-1.0, 1.0)
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
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="fp16")
    parser.add_argument("--num-frames", type=int, default=77)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--channels-last", action="store_true",
                        help="Apply channels_last_3d memory format (optional).")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile encoder/decoder (optional).")
    parser.add_argument("--skip-encode", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()

    assert (args.num_frames - 1) % 4 == 0
    assert args.height % 8 == 0 and args.width % 8 == 0

    device = torch.device("cuda")
    dtype = DTYPE_MAP[args.dtype]
    torch.set_grad_enabled(False)

    print(f"[config] backend=sglang dtype={dtype} "
          f"shape=[1,3,{args.num_frames},{args.height},{args.width}] "
          f"channels_last={args.channels_last} compile={args.compile}")

    t0 = time.time()
    vae = build_vae(dtype=dtype, device=device)
    print(f"[load] VAE ready in {time.time() - t0:.2f}s")

    if args.channels_last:
        apply_channels_last_3d(vae)

    if args.compile:
        print("[compile] compiling encoder/decoder with inductor...")
        # "default" (not "reduce-overhead") because the VAE's in-place feature
        # cache mutation conflicts with CUDA graph capture across chunk calls.
        vae.encoder = torch.compile(
            vae.encoder, backend="inductor", mode="default", fullgraph=False,
        )
        vae.decoder = torch.compile(
            vae.decoder, backend="inductor", mode="default", fullgraph=False,
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
