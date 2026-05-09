"""Profile the Wan 2.2 VAE from vllm-omni with VAE patch parallelism.

Drives `DistributedAutoencoderKLWan` directly (no DiT, no scheduler) on N GPUs
launched by torchrun. Each rank computes its spatial patch; the wrapper
gathers/stitches under the hood. NVTX markers + cudaProfilerStart/Stop gate
the nsys capture to the post-warmup measurement region.

Launch:
    torchrun --nproc-per-node=8 profile_vllm_omni.py [args...]
"""

import argparse
import os
import time

import torch
import torch._dynamo
import torch.distributed as dist
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)

# vllm-omni inherits the upstream WanResample. The default `feat_cache[idx] ==
# "Rep"` comparison makes torch.compile fullgraph=True bail out (gb0208,
# Tensor==str returns NotImplemented). Re-route via isinstance(..., str) which
# is traceable. Patched up front so any later import of the encoder/decoder
# picks up the fixed forward.
try:
    from diffusers.models.autoencoders.autoencoder_kl_wan import CACHE_T, WanResample

    def _wan_resample_forward_no_str_eq(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    is_rep = isinstance(feat_cache[idx], str)
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and not is_rep:
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                        )
                    if cache_x.shape[2] < 2 and is_rep:
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if is_rep:
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    WanResample.forward = _wan_resample_forward_no_str_eq
except Exception:
    pass

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


def apply_channels_last_3d(model: torch.nn.Module) -> torch.nn.Module:
    """Convert only the 5D parameters/buffers to channels_last_3d.

    `model.to(memory_format=torch.channels_last_3d)` errors on recent PyTorch
    when any parameter is not rank 5 (the VAE has Conv2d inside attention).
    """
    for p in model.parameters():
        if p.dim() == 5:
            p.data = p.data.to(memory_format=torch.channels_last_3d)
    for b in model.buffers():
        if b.dim() == 5:
            b.data = b.data.to(memory_format=torch.channels_last_3d)
    return model


def time_pass(fn, *args):
    dist.barrier()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn(*args)
    end.record()
    torch.cuda.synchronize()
    dist.barrier()
    return result, start.elapsed_time(end)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="fp32")
    parser.add_argument("--num-frames", type=int, default=81,
                        help="Must satisfy (N-1) % 4 == 0.")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip torch.compile (baseline run).")
    parser.add_argument("--no-channels-last", action="store_true",
                        help="Skip channels_last_3d memory format.")
    parser.add_argument("--compile-mode", default="default",
                        choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--skip-encode", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--vae-patch-parallel-size", type=int, default=8)
    parser.add_argument("--timings-dir", default=None,
                        help="Write a per-rank timings.txt here. Defaults to "
                             "$TIMINGS_DIR or the script directory.")
    args = parser.parse_args()

    assert (args.num_frames - 1) % 4 == 0, \
        f"num_frames={args.num_frames} must satisfy (N-1) % 4 == 0"
    assert args.height % 8 == 0 and args.width % 8 == 0, "H/W must be multiples of 8"

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    init_distributed_environment(backend="nccl", rank=rank, world_size=world_size)
    # DiT parallel size = DP * CFG * SP * PP * TP must equal world_size for the
    # DIT group (used by VAE patch parallel via get_dit_group) to cover all
    # ranks. We don't actually use DP for batched inference here — it's just
    # the cleanest way to put all 8 ranks in the DiT group without engaging
    # TP/SP/CFG (which are DiT-only and irrelevant to the VAE).
    initialize_model_parallel(
        data_parallel_size=world_size,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )
    is_main = rank == 0

    device = torch.device(f"cuda:{local_rank}")
    dtype = DTYPE_MAP[args.dtype]
    torch.set_grad_enabled(False)

    # End-to-end summary: rank 0 writes ONE txt with the slowest-rank per-iter
    # numbers (the true wall-clock). Per-rank detail belongs in nsys-ui — every
    # rank shows up as its own Process row in the Timeline view, with NVTX
    # decode_iter_* ranges visible per rank — so we don't duplicate it on disk.
    timings_dir = args.timings_dir or os.environ.get("TIMINGS_DIR") or os.path.dirname(
        os.path.abspath(__file__)
    )
    summary_fp = None
    if is_main:
        os.makedirs(timings_dir, exist_ok=True)
        summary_path = os.path.join(
            timings_dir,
            f"timings_pp{args.vae_patch_parallel_size}_world{world_size}_summary.txt",
        )
        summary_fp = open(summary_path, "w", buffering=1)
        summary_fp.write(
            f"# world_size={world_size} pp_size={args.vae_patch_parallel_size} "
            f"dtype={args.dtype} shape=[1,3,{args.num_frames},{args.height},{args.width}] "
            f"compile={not args.no_compile} channels_last={not args.no_channels_last}\n"
            f"# 'slow' = max over ranks per iter (true wall-clock; barriers gate to slowest)\n"
            f"# per-rank/per-GPU detail: open the .nsys-rep in nsys-ui (Timeline view)\n"
        )
    if dist.is_initialized():
        dist.barrier()

    def gather_and_summarize(phase: str, times: list[float]) -> None:
        # Each rank's `times` has length args.repeat. Gather to rank 0; rank 0
        # prints a small per-rank table to stdout (handy when watching the run)
        # and writes only the slow-row end-to-end numbers to the summary file.
        local = torch.tensor(times, device=device, dtype=torch.float64)
        gathered = (
            [torch.empty_like(local) for _ in range(world_size)] if is_main else None
        )
        dist.gather(local, gather_list=gathered, dst=0)
        if not is_main:
            return
        n = len(times)
        slowest_per_iter = [
            max(gathered[r][i].item() for r in range(world_size)) for i in range(n)
        ]
        slowest_mean = sum(slowest_per_iter) / n
        # Stdout: per-rank table for visibility while the run is happening.
        header = "rank  " + "".join(f"iter_{i:<5}".ljust(11) for i in range(n)) + "mean"
        print(f"\n=== per-rank {phase} times (ms, stdout only — see nsys for detail) ===")
        print(header)
        for r in range(world_size):
            row = gathered[r].tolist()
            mean = sum(row) / len(row)
            print(
                f"{r:<5} "
                + "".join(f"{x:<10.2f} " for x in row)
                + f"{mean:.2f}"
            )
        print(
            f"slow  "
            + "".join(f"{x:<10.2f} " for x in slowest_per_iter)
            + f"{slowest_mean:.2f}  <- max across ranks per iter (true wall-clock)"
        )
        # Summary file: only the end-to-end slow numbers.
        summary_fp.write(f"\n[{phase}] slowest-rank per iter (ms):\n")
        for i, v in enumerate(slowest_per_iter):
            summary_fp.write(f"  iter_{i}\t{v:.4f}\n")
        summary_fp.write(f"  mean\t{slowest_mean:.4f}\n")
        summary_fp.write(
            f"  min\t{min(slowest_per_iter):.4f}\n"
            f"  max\t{max(slowest_per_iter):.4f}\n"
        )
    # Same seed on every rank: each rank synthesizes the same input tensor, so
    # the patch-parallel scatter sees consistent data across the world.
    torch.manual_seed(42)

    if is_main:
        print(f"[config] world_size={world_size} pp_size={args.vae_patch_parallel_size} "
              f"dtype={dtype} shape=[1,3,{args.num_frames},{args.height},{args.width}] "
              f"compile={not args.no_compile} channels_last={not args.no_channels_last}")

    t0 = time.time()
    vae = DistributedAutoencoderKLWan.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=dtype
    ).to(device).eval()
    if is_main:
        print(f"[load] VAE loaded in {time.time() - t0:.2f}s")

    if not args.no_channels_last:
        apply_channels_last_3d(vae)
        if is_main:
            print("[config] applied channels_last_3d to 5D VAE weights")

    if not args.no_compile:
        if is_main:
            print(f"[compile] compiling encoder and decoder (mode={args.compile_mode})...")
        # Compile inner submodules before the patch-parallel wrapper takes over.
        vae.encoder = torch.compile(
            vae.encoder, backend="inductor",
            mode=args.compile_mode, fullgraph=True,
        )
        vae.decoder = torch.compile(
            vae.decoder, backend="inductor",
            mode=args.compile_mode, fullgraph=True,
        )

    vae.set_parallel_size(args.vae_patch_parallel_size)
    # Tiling is only useful when there are ≥2 ranks to distribute tiles to.
    # At pp=1, enable_tiling() + tiled_encode/tiled_decode would force the
    # 4×7=28-tile spatial loop in diffusers' parent class to run sequentially
    # on the one GPU we have, which is strictly slower than the non-tiled
    # per-chunk forward (~2.3× at 1280×720, fp32, on B200).
    if args.vae_patch_parallel_size > 1:
        vae.enable_tiling()

    latent_t = 1 + (args.num_frames - 1) // 4
    latent_h = args.height // 8
    latent_w = args.width // 8
    if is_main:
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

    def encode_one(x: torch.Tensor):
        if args.vae_patch_parallel_size > 1:
            return vae.tiled_encode(x)
        # pp=1: non-tiled per-chunk forward via diffusers' _encode (use_tiling
        # is False so the use_tiling-gated branch in _encode is skipped).
        return vae.encode(x).latent_dist.parameters

    def decode_one(z: torch.Tensor):
        if args.vae_patch_parallel_size > 1:
            return vae.tiled_decode(z)
        return vae.decode(z).sample

    # --- Encode --------------------------------------------------------------
    if not args.skip_encode:
        if is_main:
            print("\n=== ENCODE ===")
        x = make_pixel()
        for w in range(args.warmup):
            t0 = time.time()
            _, ms = time_pass(encode_one, x)
            if is_main:
                print(f"[encode warmup {w}] {ms:.2f} ms (wall {time.time() - t0:.2f}s)")

        torch.cuda.reset_peak_memory_stats()
        cuda_profiler_start()
        nvtx_push("encode_measure")
        encode_times = []
        for i in range(args.repeat):
            nvtx_push(f"encode_iter_{i}")
            _, ms = time_pass(encode_one, x)
            nvtx_pop()
            encode_times.append(ms)
        nvtx_pop()
        cuda_profiler_stop()

        gather_and_summarize("encode", encode_times)
        if is_main:
            print(
                f"[encode] peak_alloc={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB "
                f"peak_reserved={torch.cuda.max_memory_reserved() / 2**30:.2f} GiB"
            )
        del x

    # --- Decode --------------------------------------------------------------
    if not args.skip_decode:
        if is_main:
            print("\n=== DECODE ===")
        z = make_latent()
        for w in range(args.warmup):
            t0 = time.time()
            _, ms = time_pass(decode_one, z)
            if is_main:
                print(f"[decode warmup {w}] {ms:.2f} ms (wall {time.time() - t0:.2f}s)")

        torch.cuda.reset_peak_memory_stats()
        cuda_profiler_start()
        nvtx_push("decode_measure")
        decode_times = []
        for i in range(args.repeat):
            nvtx_push(f"decode_iter_{i}")
            _, ms = time_pass(decode_one, z)
            nvtx_pop()
            decode_times.append(ms)
        nvtx_pop()
        cuda_profiler_stop()

        gather_and_summarize("decode", decode_times)
        if is_main:
            print(
                f"[decode] peak_alloc={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB "
                f"peak_reserved={torch.cuda.max_memory_reserved() / 2**30:.2f} GiB"
            )

    if is_main and summary_fp is not None:
        summary_fp.close()
        print(f"[timings] end-to-end summary: {summary_path}")
        print(f"[timings] per-rank/per-GPU detail: open the .nsys-rep in nsys-ui")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
