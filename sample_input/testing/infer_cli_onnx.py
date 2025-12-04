# infer_cli_onnx.py
import os, json, shlex, subprocess, argparse
import numpy as np
import torch
import onnxruntime as ort
import shutil
import cv2
import matplotlib
matplotlib.use("Agg")  # headless save on servers/CI/WSL
import matplotlib.pyplot as plt
import random  # <-- add this



# ---- same normalization as training ----
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
IMN_MEAN_5D = IM_MEAN.view(1,1,3,1,1)
IMN_STD_5D  = IM_STD.view(1,1,3,1,1)


IMN_MEAN_4D = IM_MEAN            # [1,3,1,1]
IMN_STD_4D  = IM_STD             # [1,3,1,1]


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

    # NEW: detect raw .rgb input
    is_raw_rgb = src_video.lower().endswith(".rgb")

    if os.path.exists(raw_path) and not overwrite:
        N = finalize_memmap(raw_path, cache_dir, size, src_video, keep_every, roi)
        return cache_dir, N

    if is_raw_rgb:
        # Just copy the raw RGB file into our cache location.
        # Assumes frames are already [N, size, size, 3] uint8.
        if os.path.abspath(src_video) != os.path.abspath(raw_path):
            shutil.copyfile(src_video, raw_path)
    else:
        # Original FFmpeg extraction path for mp4 / other video formats.
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


def psnr_luma(a01: torch.Tensor, b01: torch.Tensor, eps=1e-12) -> torch.Tensor:
    # a01,b01: [B,3,H,W] in [0,1]
    Y1 = 0.299*a01[:,0:1] + 0.587*a01[:,1:2] + 0.114*a01[:,2:3]
    Y2 = 0.299*b01[:,0:1] + 0.587*b01[:,1:2] + 0.114*b01[:,2:3]
    mse_y = torch.mean((Y1 - Y2) ** 2, dim=(1,2,3)).clamp_min(eps)
    return 10.0 * torch.log10(1.0 / mse_y)

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
def _mean_std_chw(chw: torch.Tensor):
    # chw: [3,H,W], normalized; returns ( (mr,mg,mb), (sr,sg,sb) )
    c = chw.reshape(3, -1)
    m = c.mean(dim=1)
    s = c.std(dim=1, unbiased=False)
    return m.tolist(), s.tolist()


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
    ap.add_argument("--stat_every", type=int, default=1,
                help="If >0, log per-frame mean/std every Nth window (normalized space).")
    ap.add_argument("--stat_samples", type=int, default=1,
                help="How many batch samples to log per matching window.")
    ap.add_argument(
        "--save_error_maps", type=int, default=0, help="If >0, save this many error-map images (|pred-target|) for debugging.",)

    args = ap.parse_args()

    # ROI parse
    roi = None
    if args.roi:
        xs = [int(v) for v in args.roi.split(",")]
        if len(xs) != 4: raise SystemExit("--roi must be 'x,y,w,h'")
        roi = tuple(xs)

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

    # how many target frames we’ll score: one target per sliding window
    num_targets = max(0, N - seq_len)  # windows = N - seq_len

    # existing fixed cap for pred|target debug pairs
    save_cap = 60
    saved_pairs = 0

    # NEW: random sampling probabilities
    if num_targets > 0:
        p_pair = min(1.0, save_cap / num_targets)
        p_err = 0.0
        if getattr(args, "save_error_maps", 0) > 0:
            p_err = min(1.0, args.save_error_maps / num_targets)
    else:
        p_pair = 0.0
        p_err = 0.0

    saved_errmaps = 0  # NEW: track how many error maps we’ve written

    T = seq_len + 1  # 4-in + 1-target for scoring
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    mean5 = IMN_MEAN_5D.to(device); std5 = IMN_STD_5D.to(device)

    all_starts, all_psnr_rgb, all_psnr_luma = [], [], []

    with torch.no_grad():
        for b_u8, starts in memmap_window_batch_iter(cache_dir, T=T, batch_size=args.batch_size, return_indices=True):
            if device == "cuda":
                b_u8 = b_u8.pin_memory()
            b01 = b_u8.to(device=device, dtype=torch.float32, non_blocking=(device=="cuda")).div_(255.0)

            b_in  = b01[:, :seq_len]   # [B,seq,3,H,W]
            tgt01 = b01[:,  seq_len]   # [B,3,H,W]

            # Normalize like training
            x_norm = (b_in - mean5) / std5  # [B,seq,3,H,W]

            # -------- stats logging (normalized space) --------
            if getattr(args, "stat_every", 0):
                se = int(args.stat_every)
                if se > 0:
                    samples_to_log = min(int(getattr(args, "stat_samples", 1)), x_norm.shape[0])
                    for bi in range(samples_to_log):
                        start_idx = int(starts[bi])
                        if (start_idx % se) != (se - 1):
                            continue

                        # inputs per time step
                        xin = x_norm[bi]  # [seq,3,H,W]
                        for t in range(xin.shape[0]):
                            m, s = _mean_std_chw(xin[t])
                            print(
                                f"PY win={start_idx} t={t} input.norm "
                                f"mean=[{m[0]:.4f} {m[1]:.4f} {m[2]:.4f}] "
                                f"std=[{s[0]:.4f} {s[1]:.4f} {s[2]:.4f}]"
                            )

            # -------- ONNX run (per-sample; model has no batch axis) --------
            preds = []
            for i in range(x_norm.shape[0]):
                xin = x_norm[i]  # [seq,3,H,W]
                np_in = (xin.half() if use_fp16 else xin.float()).cpu().numpy()
                np_in = np.ascontiguousarray(np_in)
                out = sess.run([output_name], {input_name: np_in})[0]  # [3,H,W]
                y_norm = torch.from_numpy(out).to(device=device, dtype=torch.float32)  # normalized space
                preds.append(y_norm)
            y_norm_b = torch.stack(preds, dim=0)   # [B,3,H,W] normalized

            # -------- more stats: pred + target (normalized) --------
            if getattr(args, "stat_every", 0):
                se = int(args.stat_every)
                if se > 0:
                    samples_to_log = min(int(getattr(args, "stat_samples", 1)), y_norm_b.shape[0])
                    for bi in range(samples_to_log):
                        start_idx = int(starts[bi])
                        if (start_idx % se) != (se - 1):
                            continue

                        # prediction (normalized)
                        pm, ps = _mean_std_chw(y_norm_b[bi])
                        print(
                            f"PY win={start_idx} pred.norm "
                            f"mean=[{pm[0]:.4f} {pm[1]:.4f} {pm[2]:.4f}] "
                            f"std=[{ps[0]:.4f} {ps[1]:.4f} {ps[2]:.4f}]"
                        )

                        # target normalized
                        tgt_norm = (tgt01[bi:bi+1] - IMN_MEAN_4D.to(device)) / IMN_STD_4D.to(device)  # [1,3,H,W]
                        tm, ts = _mean_std_chw(tgt_norm[0])
                        tgt_idx = start_idx + seq_len
                        print(
                            f"PY win={start_idx} tgt={tgt_idx} target.norm "
                            f"mean=[{tm[0]:.4f} {tm[1]:.4f} {tm[2]:.4f}] "
                            f"std=[{ts[0]:.4f} {ts[1]:.4f} {ts[2]:.4f}]"
                        )

            # -------- metrics on [0,1] --------
            pred01 = denorm01(y_norm_b).clamp(0,1)
            _psnr_rgb  = psnr_rgb(pred01, tgt01)   # [B]
            _psnr_y    = psnr_luma(pred01, tgt01)  # [B]

            all_starts.extend(starts.tolist())
            all_psnr_rgb.extend(_psnr_rgb.detach().cpu().tolist())
            all_psnr_luma.extend(_psnr_y.detach().cpu().tolist())
            # choose up to a few samples per batch for visual debug / error maps
            k = min(3, pred01.shape[0])  # don’t explode disk

            # --- randomized debug saving over all frames, not just the first ones ---
            # one “target frame” per window/sample in this batch
            for i in range(pred01.shape[0]):
                ti = int(starts[i]) + seq_len  # global target frame index

                # decide randomly whether to save pred|target pair for this frame
                if saved_pairs < save_cap and random.random() < p_pair:
                    pred_u8 = (pred01[i].clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy())  # [3,H,W]
                    tgt_u8  = (tgt01[i].clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy())   # [3,H,W]

                    pred_bgr = cv2.cvtColor(np.transpose(pred_u8, (1, 2, 0)), cv2.COLOR_RGB2BGR)
                    tgt_bgr  = cv2.cvtColor(np.transpose(tgt_u8,  (1, 2, 0)), cv2.COLOR_RGB2BGR)
                    side = cv2.hconcat([pred_bgr, tgt_bgr])

                    cv2.imwrite(os.path.join(args.out_dir, f"debug_pair_t{ti:06d}.jpg"), side)
                    saved_pairs += 1

                # decide randomly whether to save an error map for this frame
                if getattr(args, "save_error_maps", 0) > 0 and saved_errmaps < args.save_error_maps and random.random() < p_err:
                    # per-pixel mean abs error across channels -> [H,W] in [0,1]
                    err = (pred01[i] - tgt01[i]).abs().mean(dim=0)  # [H,W]
                    max_val = err.max()
                    if max_val > 0:
                        err = err / max_val  # normalize for visualization

                    err_u8 = err.mul(255).to(torch.uint8).cpu().numpy()  # [H,W]
                    err_color = cv2.applyColorMap(err_u8, cv2.COLORMAP_JET)

                    cv2.imwrite(os.path.join(args.out_dir, f"error_map_t{ti:06d}.jpg"), err_color)
                    saved_errmaps += 1


    # write CSV
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    with open(csv_path, "w") as f:
        f.write("start_idx,target_idx,psnr_rgb,psnr_luma\n")
        for s, p_rgb, p_y in zip(all_starts, all_psnr_rgb, all_psnr_luma):
            f.write(f"{s},{s+seq_len},{p_rgb:.6f},{p_y:.6f}\n")
    print(f"[infer] wrote {csv_path}  rows={len(all_starts)}")
    # --- plots next to CSV ---
    try:
        # x-axis: target frame index
        x = np.array([s + seq_len for s in all_starts], dtype=np.int64)
        psnr = np.array(all_psnr_rgb, dtype=np.float32)
        lmse = np.array(all_psnr_luma, dtype=np.float32)

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
        np.save(os.path.join(args.out_dir, "psnr_rgb.npy"), np.array(all_psnr_rgb, dtype=np.float32))
        np.save(os.path.join(args.out_dir, "luma_mse.npy"), np.array(all_psnr_luma, dtype=np.float32))

if __name__ == "__main__":
    main()
