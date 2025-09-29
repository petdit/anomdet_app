# infer_cli.py
import os, json, shlex, subprocess, argparse
import numpy as np
import torch
import cv2
from math import log10

# ---- same normalization as training ----
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
IMN_MEAN_5D = IM_MEAN.view(1,1,3,1,1)
IMN_STD_5D  = IM_STD.view(1,1,3,1,1)

def denorm01(x):  # x normalized -> [0,1]
    m, s = IM_MEAN.to(x.device), IM_STD.to(x.device)
    return (x * s + m).clamp(0, 1)

def default_cache_root():
    return os.environ.get("FRAMECACHE_DIR", "/opt/ml/.cache/frames")

# ---- FFmpeg filter: exact stepping -> ROI -> square crop (wide=center, tall=bottom) -> scale -> rgb24 ----
def build_vf_filter(size: int, roi, keep_every: int) -> str:
    parts = []
    if keep_every and keep_every > 1:
        parts.append(f"select='not(mod(n,{int(keep_every)}))',setpts=N/FRAME_RATE/TB")
    if roi is not None:
        x,y,w,h = roi
        parts.append(f"crop={w}:{h}:{x}:{y}")
    parts.append(
        "crop="
        "w='if(gte(iw,ih),ih,iw)':"
        "h='if(gte(iw,ih),ih,iw)':"
        "x='if(gte(iw,ih),(iw-ih)/2,0)':"  # wide -> center horizontally
        "y='if(gte(iw,ih),0,ih-iw)'"       # tall -> bottom crop
    )
    parts.append(f"scale={size}:{size}:flags=area")
    parts.append("format=rgb24")
    return ",".join(parts)

def run_ffmpeg_to_raw(src_video: str, vf: str, threads: int, raw_path: str):
    cmd = (f'ffmpeg -hide_banner -loglevel error -nostdin '
           f'-threads {threads} -i "{src_video}" -an -sn -dn '
           f'-vf "{vf}" -f rawvideo -pix_fmt rgb24 -vsync vfr -y -')
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "wb") as f:
        proc = subprocess.Popen(shlex.split(cmd), stdout=f)
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg exited with code {ret}")

def finalize_memmap(raw_path: str, dst_dir: str, size: int, src_video: str, keep_every, roi):
    bpf = size * size * 3
    total = os.path.getsize(raw_path)
    if total % bpf != 0:
        raise ValueError(f"Raw size {total} not divisible by frame size {bpf}")
    N = total // bpf
    meta = {
        "src_video": src_video,
        "shape": {"N": int(N), "H": size, "W": size, "C": 3, "layout": "NHWC", "dtype": "uint8"},
        "filters": {"keep_every": keep_every or 1, "roi": roi, "scale": size, "pix_fmt": "rgb24"}
    }
    with open(os.path.join(dst_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return int(N)

def extract_to_memmap(src_video: str, cache_root: str, *, size=224, roi=None,
                      keep_every=None, threads=0, overwrite=False) -> tuple[str, int]:
    base = os.path.splitext(os.path.basename(src_video))[0]
    tag  = f"size{size}__{'roi' if roi else 'auto'}"
    if roi: tag += f"_roi_{roi[0]}_{roi[1]}_{roi[2]}_{roi[3]}"
    if keep_every and keep_every > 1: tag += f"__keep{int(keep_every)}"
    cache_dir = os.path.join(cache_root, f"{base}__{tag}")
    os.makedirs(cache_dir, exist_ok=True)
    raw_path = os.path.join(cache_dir, "frames.rgb")
    if os.path.exists(raw_path) and not overwrite:
        N = finalize_memmap(raw_path, cache_dir, size, src_video, keep_every, roi)
        return cache_dir, N
    vf = build_vf_filter(size=size, roi=roi, keep_every=keep_every)
    run_ffmpeg_to_raw(src_video, vf, threads, raw_path)
    N = finalize_memmap(raw_path, cache_dir, size, src_video, keep_every, roi)
    return cache_dir, N

def open_memmap_cache(cache_dir: str):
    meta = json.load(open(os.path.join(cache_dir, "metadata.json"), "r"))
    N,H,W,C = meta["shape"]["N"], meta["shape"]["H"], meta["shape"]["W"], meta["shape"]["C"]
    arr = np.memmap(os.path.join(cache_dir, "frames.rgb"), mode="r", dtype=np.uint8, shape=(N,H,W,C))
    return arr, meta

def memmap_window_batch_iter(cache_dir: str, *, T: int, batch_size: int, return_indices: bool = False):
    arr, _ = open_memmap_cache(cache_dir)
    N = arr.shape[0]
    starts = list(range(0, max(0, N - T + 1)))
    batch, idxs = [], []
    for s in starts:
        win = arr[s:s+T]                    # [T,H,W,3]
        win = np.transpose(win, (0,3,1,2))  # [T,3,H,W]
        batch.append(win); idxs.append(s)
        if len(batch) == batch_size:
            bt = torch.from_numpy(np.stack(batch, axis=0))  # [B,T,3,H,W] u8
            if return_indices: yield bt, np.array(idxs, dtype=np.int64)
            else:              yield bt
            batch, idxs = [], []
    if batch:
        bt = torch.from_numpy(np.stack(batch, axis=0))
        if return_indices: yield bt, np.array(idxs, dtype=np.int64)
        else:              yield bt

# ---- metrics: PSNR (RGB on [0,1]) and luma-MSE ----
def luma(x01: torch.Tensor) -> torch.Tensor:
    return 0.299*x01[:,0:1] + 0.587*x01[:,1:2] + 0.114*x01[:,2:3]

def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2, dim=(1,2,3))  # per-sample

def psnr_rgb(a01: torch.Tensor, b01: torch.Tensor, eps=1e-12) -> torch.Tensor:
    # PSNR = 10 * log10(MAX^2 / MSE). For [0,1], MAX=1.
    per = torch.mean((a01 - b01) ** 2, dim=(1,2,3)).clamp_min(eps)
    return 10.0 * torch.log10(1.0 / per)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to model.ts.pt (TorchScript)")
    ap.add_argument("--video", required=True, help="Path to input .mp4")
    ap.add_argument("--out_dir", required=True, help="Where to write outputs")
    ap.add_argument("--seq_len", type=int, default=4, help="Number of input frames")
    ap.add_argument("--size", type=int, default=224, help="Square resize dim")
    ap.add_argument("--keep_every", type=int, default=None, help="Exact frame step (e.g., 2 keeps every 2nd)")
    ap.add_argument("--roi", type=str, default=None, help="ROI as 'x,y,w,h' or omit")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0, help="FFmpeg threads (0=auto)")
    ap.add_argument("--overwrite_cache", action="store_true")
    ap.add_argument("--save_npy", action="store_true", help="Also save psnr.npy and luma_mse.npy")
    args = ap.parse_args()

    roi = None
    if args.roi:
        xs = [int(v) for v in args.roi.split(",")]
        if len(xs) != 4: raise SystemExit("--roi must be 'x,y,w,h'")
        roi = tuple(xs)

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(args.model, map_location=device).eval()

    cache_dir, N = extract_to_memmap(
        src_video=args.video,
        cache_root=default_cache_root(),
        size=args.size, roi=roi,
        keep_every=args.keep_every,
        threads=args.threads,
        overwrite=args.overwrite_cache
    )
    print(f"[infer] cache={cache_dir}  frames={N}")

    T = args.seq_len + 1  # 4-in + 1-target for scoring
    mean5 = IMN_MEAN_5D.to(device); std5 = IMN_STD_5D.to(device)

    all_starts, all_psnr, all_luma_mse = [], [], []

    with torch.no_grad():
        for b_u8, starts in memmap_window_batch_iter(cache_dir, T=T, batch_size=args.batch_size, return_indices=True):
            if device == "cuda":
                b_u8 = b_u8.pin_memory()
            b01 = b_u8.to(device=device, dtype=torch.float32, non_blocking=(device=="cuda")).div_(255.0)

            b_in  = b01[:, :args.seq_len]    # [B,seq,3,H,W], no aug
            tgt01 = b01[:,  args.seq_len]    # [B,3,H,W]

            x = (b_in - mean5) / std5
            y = model(x)                     # [B,3,H,W] normalized
            pred01 = denorm01(y)             # [B,3,H,W] in [0,1]

            # metrics
            _psnr = psnr_rgb(pred01, tgt01)                    # [B]
            _lmse = mse(luma(pred01), luma(tgt01))             # [B]

            all_starts.extend(starts.tolist())
            all_psnr.extend(_psnr.detach().cpu().tolist())
            all_luma_mse.extend(_lmse.detach().cpu().tolist())

    # write CSV
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    with open(csv_path, "w") as f:
        f.write("start_idx,target_idx,psnr_rgb,luma_mse\n")
        for s, p, m in zip(all_starts, all_psnr, all_luma_mse):
            f.write(f"{s},{s+args.seq_len},{p:.6f},{m:.8f}\n")
    print(f"[infer] wrote {csv_path}  rows={len(all_starts)}")

    if args.save_npy:
        np.save(os.path.join(args.out_dir, "psnr_rgb.npy"), np.array(all_psnr, dtype=np.float32))
        np.save(os.path.join(args.out_dir, "luma_mse.npy"), np.array(all_luma_mse, dtype=np.float32))

if __name__ == "__main__":
    main()
