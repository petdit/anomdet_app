# infer_cli_onnx.py
import os, json, shlex, subprocess, argparse
import numpy as np
import torch
import onnxruntime as ort
import cv2
import matplotlib
matplotlib.use("Agg")  # headless save on servers/CI/WSL
import matplotlib.pyplot as plt
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
        "x='if(gte(iw,ih),(iw-ih)/2,0)':"
        "y='if(gte(iw,ih),0,ih-iw)'"
    )
    parts.append(f"scale={size}:{size}:flags=area")
    parts.append("format=rgb24,setsar=1")
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
    per = torch.mean((a01 - b01) ** 2, dim=(1,2,3)).clamp_min(eps)
    return 10.0 * torch.log10(1.0 / per)

def _load_export_metadata(onnx_path: str):
    """Load metadata.json next to the model if present; return dict (or {})."""
    d = os.path.dirname(os.path.abspath(onnx_path))
    mpath = os.path.join(d, "metadata.json")
    if os.path.exists(mpath):
        try:
            with open(mpath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _pick_providers(prefer_cuda: bool = True):
    avail = set(ort.get_available_providers())
    order = []
    if prefer_cuda and "CUDAExecutionProvider" in avail:
        order.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in avail:
        order.append("CPUExecutionProvider")
    if not order:
        order = list(avail)
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to model.onnx or model.ort")
    ap.add_argument("--video", required=True, help="Path to input .mp4")
    ap.add_argument("--out_dir", required=True, help="Where to write outputs")
    ap.add_argument("--seq_len", type=int, default=None, help="Number of input frames (defaults from metadata)")
    ap.add_argument("--size", type=int, default=None, help="Square resize dim (defaults from metadata)")
    ap.add_argument("--keep_every", type=int, default=None, help="Exact frame step (e.g., 2 keeps every 2nd)")
    ap.add_argument("--roi", type=str, default=None, help="ROI as 'x,y,w,h' or omit")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0, help="FFmpeg threads (0=auto)")
    ap.add_argument("--overwrite_cache", action="store_true")
    ap.add_argument("--save_npy", action="store_true", help="Also save psnr.npy and luma_mse.npy")
    ap.add_argument("--fp16", action="store_true", help="Force FP16 input to match an fp16-exported model")
    ap.add_argument("--cpu", action="store_true", help="Force CPUExecutionProvider")
    args = ap.parse_args()

    # ROI parse
    roi = None
    if args.roi:
        xs = [int(v) for v in args.roi.split(",")]
        if len(xs) != 4: raise SystemExit("--roi must be 'x,y,w,h'")
        roi = tuple(xs)
    saved_pairs = 0
    save_cap = 60   # how many early pairs to save

    os.makedirs(args.out_dir, exist_ok=True)

    # Infer defaults from metadata.json if present
    meta = _load_export_metadata(args.model)
    seq_len_default = int(meta.get("sequence_length", meta.get("model", {}).get("sequence_length", 4)))
    in_size_meta = meta.get("input_size", meta.get("model", {}).get("input_size", [3, 224, 224]))
    size_default = int(in_size_meta[1]) if isinstance(in_size_meta, (list, tuple)) and len(in_size_meta) == 3 else 224

    seq_len = int(args.seq_len or seq_len_default)
    size = int(args.size or size_default)

    # Precision: auto-detect from metadata.converter.convert_precision
    conv = meta.get("converter", {}) if isinstance(meta, dict) else {}
    convert_precision = str(conv.get("convert_precision", "auto")).strip().lower()
    use_fp16 = bool(args.fp16 or convert_precision in {"16", "fp16", "half"})

    # ONNX Runtime session
    providers = ["CPUExecutionProvider"] if args.cpu else _pick_providers()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # NEW
    sess = ort.InferenceSession(args.model, sess_options=so, providers=providers)
    # Expect input named "input" from export code
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    input_type = getattr(sess.get_inputs()[0], "type", "")  # e.g. 'tensor(float16)'
    if not args.fp16 and convert_precision == "auto":
        if isinstance(input_type, str) and "float16" in input_type:
            use_fp16 = True

    # Build / open frame cache
    cache_dir, N = extract_to_memmap(
        src_video=args.video,
        cache_root=default_cache_root(),
        size=size, roi=roi,
        keep_every=args.keep_every,
        threads=args.threads,
        overwrite=args.overwrite_cache
    )
    actual_providers = sess.get_providers()
    print(f"[infer] cache={cache_dir}  frames={N}  seq_len={seq_len}  size={size}  "
          f"FP16={use_fp16}  providers_requested={providers}  providers_in_use={actual_providers}  onnx_input_type={input_type}")  # NEW


    T = seq_len + 1  # 4-in + 1-target for scoring
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    mean5 = IMN_MEAN_5D.to(device); std5 = IMN_STD_5D.to(device)

    all_starts, all_psnr, all_luma_mse = [], [], []

    # Inference loop:
    # Your ONNX takes [T,3,S,S] and returns [3,S,S]; no batch dim -> run per-sample window.
    with torch.no_grad():
        for b_u8, starts in memmap_window_batch_iter(cache_dir, T=T, batch_size=args.batch_size, return_indices=True):
            if device == "cuda":
                b_u8 = b_u8.pin_memory()
            b01 = b_u8.to(device=device, dtype=torch.float32, non_blocking=(device=="cuda")).div_(255.0)

            b_in  = b01[:, :seq_len]   # [B,seq,3,H,W], no aug
            tgt01 = b01[:,  seq_len]   # [B,3,H,W]

            # Normalize like training
            x_norm = (b_in - mean5) / std5  # torch tensor [B,seq,3,H,W]

            # Run ONNX per sample (no batch axis in model)
            preds = []
            for i in range(x_norm.shape[0]):
                xin = x_norm[i]  # [seq,3,H,W]
                np_in = (xin.half() if use_fp16 else xin.float()).cpu().numpy()
                np_in = np.ascontiguousarray(np_in)  # NEW
                out = sess.run([output_name], {input_name: np_in})[0]  # [3,H,W], dtype fp16/fp32
                # convert to torch for metrics; keep on CPU then move
                y_norm = torch.from_numpy(out).to(device=device, dtype=torch.float32)  # promote to fp32 for metrics
                preds.append(y_norm)
            y_norm_b = torch.stack(preds, dim=0)   # [B,3,H,W] normalized

            pred01 = denorm01(y_norm_b)            # [B,3,H,W] in [0,1]

            # metrics
            _psnr = psnr_rgb(pred01, tgt01)                    # [B]
            _lmse = mse(luma(pred01), luma(tgt01))             # [B]

            all_starts.extend(starts.tolist())
            all_psnr.extend(_psnr.detach().cpu().tolist())
            all_luma_mse.extend(_lmse.detach().cpu().tolist())
            if saved_pairs < save_cap:
            # save a grid for the first few samples of *this* batch (typically 1 is enough)
                k = min(3, pred01.shape[0])  # don’t explode disk
                for i in range(k):
                    ti = int(starts[i]) + seq_len
                    pred_u8 = (pred01[i].clamp(0,1).mul(255).to(torch.uint8).cpu().numpy())   # [3,H,W]
                    tgt_u8  = (tgt01[i].clamp(0,1).mul(255).to(torch.uint8).cpu().numpy())    # [3,H,W]
                    pred_bgr = cv2.cvtColor(np.transpose(pred_u8, (1,2,0)), cv2.COLOR_RGB2BGR)
                    tgt_bgr  = cv2.cvtColor(np.transpose(tgt_u8,  (1,2,0)), cv2.COLOR_RGB2BGR)
                    side = cv2.hconcat([pred_bgr, tgt_bgr])
                    cv2.imwrite(os.path.join(args.out_dir, f"early_pair_t{ti:06d}.jpg"), side)
                    saved_pairs += 1
                    if saved_pairs >= save_cap:
                        break
    # write CSV
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    with open(csv_path, "w") as f:
        f.write("start_idx,target_idx,psnr_rgb,luma_mse\n")
        for s, p, m in zip(all_starts, all_psnr, all_luma_mse):
            f.write(f"{s},{s+seq_len},{p:.6f},{m:.8f}\n")
    print(f"[infer] wrote {csv_path}  rows={len(all_starts)}")
    # --- plots next to CSV ---
    try:
        # x-axis: target frame index
        x = np.array([s + seq_len for s in all_starts], dtype=np.int64)
        psnr = np.array(all_psnr, dtype=np.float32)
        lmse = np.array(all_luma_mse, dtype=np.float32)

        # rolling mean helper
        def rolling_mean(y, w: int):
            w = max(1, int(w))
            if y.size < w:
                return y
            k = np.ones(w, dtype=np.float32) / w
            return np.convolve(y, k, mode="same")

        # choose a sensible window (about ~1% of points, min 25, max 301)
        win = 1

        # PSNR plot
        plt.figure(figsize=(10, 5))
        plt.plot(x, psnr, linewidth=1.0, label="PSNR (dB)")
        plt.plot(x, rolling_mean(psnr, win), linewidth=2.0, label=f"Rolling mean (w={win})")
        plt.xlabel("target_idx")
        plt.ylabel("PSNR (dB)")
        plt.title("PSNR vs frame")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "psnr.png"), dpi=144)
        plt.close()

        # Luma-MSE plot
        plt.figure(figsize=(10, 5))
        plt.plot(x, lmse, linewidth=1.0, label="luma MSE")
        plt.plot(x, rolling_mean(lmse, win), linewidth=2.0, label=f"Rolling mean (w={win})")
        plt.xlabel("target_idx")
        plt.ylabel("MSE")
        plt.title("Luma-MSE vs frame")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "luma_mse.png"), dpi=144)
        plt.close()

        print(f"[infer] wrote plots: psnr.png, luma_mse.png")
    except Exception as e:
        print(f"[warn] failed to write plots: {e}")

    if args.save_npy:
        np.save(os.path.join(args.out_dir, "psnr_rgb.npy"), np.array(all_psnr, dtype=np.float32))
        np.save(os.path.join(args.out_dir, "luma_mse.npy"), np.array(all_luma_mse, dtype=np.float32))

if __name__ == "__main__":
    main()
