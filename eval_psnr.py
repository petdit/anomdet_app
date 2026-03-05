#!/usr/bin/env python3
"""
eval_psnr.py  –  Offline PSNR evaluation, matching Android FilmTheFault inference.

Loads a trained model (PyTorch .pth or exported ONNX) and runs it on either:
  • An MP4 video                (--video PATH)
  • Phone .bin window files     (--frames-dir PATH  with window_XXXXXXX.bin files)
  • A sorted PNG/JPG folder     (--frames-dir PATH  with image files)

Output CSV: window_start,psnr_db

PSNR is computed on the luma (BT.601) channel to match Android psnrLuma01().

Usage examples
--------------
# FP32 PyTorch, video:
python eval_psnr.py --weights run/model_final.pth --video test.mp4

# FP16 PyTorch, phone bin frames, save to custom file:
python eval_psnr.py --weights run/model_final.pth --frames-dir /tmp/frames --fp16 --out psnr_fp16.csv

# ONNX runtime, video:
python eval_psnr.py --onnx run/model.onnx --video test.mp4 --out psnr_onnx.csv
"""

import argparse
import csv
import glob
import math
import os
import sys
import tempfile
import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Normalization constants — must match trainer_light.py
# ---------------------------------------------------------------------------
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

_MEAN_T = torch.from_numpy(_MEAN)  # [3,1,1] — broadcast over H,W
_STD_T  = torch.from_numpy(_STD)


def _norm_np(x_chw_01: np.ndarray) -> np.ndarray:
    """[3,H,W] float32 [0,1] → normalized."""
    return (x_chw_01 - _MEAN) / _STD


def _denorm_t(x_norm: torch.Tensor) -> torch.Tensor:
    """[*,3,H,W] normalized → [0,1] (approximately)."""
    m = _MEAN_T.to(x_norm.device)
    s = _STD_T.to(x_norm.device)
    return x_norm * s + m


def _luma_np(x_chw_01: np.ndarray) -> np.ndarray:
    """[3,H,W] [0,1] → [H,W] luma BT.601."""
    return 0.299 * x_chw_01[0] + 0.587 * x_chw_01[1] + 0.114 * x_chw_01[2]


def _psnr_luma_np(pred_01: np.ndarray, tgt_01: np.ndarray) -> float:
    """Both [3,H,W] float [0,1]. Returns PSNR dB (luma channel, matches Android)."""
    lp = _luma_np(np.clip(pred_01, 0, 1))
    lt = _luma_np(np.clip(tgt_01,  0, 1))
    mse = np.mean((lp - lt) ** 2)
    return float(10 * math.log10(1.0 / (mse + 1e-8)))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_pytorch(weights_path: str, seq_len: int, fp16: bool, device: str):
    """Load CNN_TSM_NNAPI from a .pth state-dict, return model in eval mode."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model_min import CNN_TSM_NNAPI

    model = CNN_TSM_NNAPI(sequence_length=seq_len)
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    if fp16:
        model = model.half()
    return model.to(device)


def _load_onnx(onnx_path: str):
    """Load ONNX model via onnxruntime, return (session, input_name, output_name)."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("onnxruntime not installed. Run: pip install onnxruntime")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    print(f"[onnx] input={in_name!r} shape={sess.get_inputs()[0].shape}  "
          f"output={out_name!r} shape={sess.get_outputs()[0].shape}")
    return sess, in_name, out_name


# ---------------------------------------------------------------------------
# Single-window inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def _infer_pytorch_window(model, inp_tchw_u8: np.ndarray,
                          fp16: bool, device: str) -> np.ndarray:
    """
    inp_tchw_u8: [T, 3, H, W] uint8 — the T input frames.
    Returns predicted frame [3, H, W] float32 in normalized space.
    """
    x = inp_tchw_u8.astype(np.float32) / 255.0       # [T, 3, H, W] [0,1]
    x = (x - _MEAN) / _STD                            # normalized
    t = torch.from_numpy(x).to(device)
    if fp16:
        t = t.half()
    out = model(t)                                     # [3, H, W] normalized
    return out.float().cpu().numpy()


def _infer_onnx_window(sess, in_name: str, out_name: str,
                       inp_tchw_u8: np.ndarray) -> np.ndarray:
    """
    inp_tchw_u8: [T, 3, H, W] uint8.
    Returns predicted frame [3, H, W] float32 in normalized space.
    ONNX model expects [T, 3, H, W] float32 normalized input.
    """
    x = inp_tchw_u8.astype(np.float32) / 255.0        # [T, 3, H, W] [0,1]
    x_norm = (x - _MEAN) / _STD                       # [T, 3, H, W] normalized
    pred = sess.run([out_name], {in_name: x_norm})[0]  # [3, H, W] normalized
    return pred.astype(np.float32)


def _pred_to_01(pred_norm: np.ndarray) -> np.ndarray:
    """Denormalize [3,H,W] normalized → [3,H,W] float [0,1]."""
    return np.clip(pred_norm * _STD + _MEAN, 0, 1)


# ---------------------------------------------------------------------------
# Data iterators  →  yield (starts: list[int], windows: np.ndarray [B,T+1,3,H,W] uint8)
# ---------------------------------------------------------------------------

def _iter_video(video_path: str, seq_len: int, batch_size: int,
                size: int, frame_stride: int, n_windows=None):
    """Extract video to temp memmap then iterate windows in order."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from trainer_light import extract_to_memmap, open_memmap_cache, window_indices

    cache_root = tempfile.mkdtemp(prefix="eval_psnr_")
    print(f"[video] Extracting frames: {video_path}  → cache {cache_root}")
    cache_dir, N = extract_to_memmap(video_path, cache_root, size=size)
    arr, _ = open_memmap_cache(cache_dir)   # [N, H, W, 3] uint8

    T_total = seq_len + 1       # input frames + target
    S = max(1, int(frame_stride))
    max_start = N - (T_total - 1) * S
    starts_all = list(range(0, max(0, max_start)))
    if n_windows is not None:
        starts_all = starts_all[:n_windows]

    batch_wins, batch_starts = [], []
    for s in starts_all:
        idx = np.arange(s, s + T_total * S, S, dtype=np.int64)   # [T_total]
        win = arr[idx]                                             # [T_total, H, W, 3]
        win = np.transpose(win, (0, 3, 1, 2))                     # [T_total, 3, H, W]
        batch_wins.append(win)
        batch_starts.append(s)
        if len(batch_wins) == batch_size:
            yield batch_starts, np.stack(batch_wins, axis=0)
            batch_wins, batch_starts = [], []
    if batch_wins:
        yield batch_starts, np.stack(batch_wins, axis=0)


def _iter_bin(frames_dir: str, seq_len: int, batch_size: int, n_windows=None):
    """
    Iterate phone window_XXXXXXX.bin files in sorted (chronological) order.
    Each file = (seq_len+1) × H × W × 3 bytes, NHWC uint8.
    Yields (starts, windows [B, T+1, 3, H, W]).
    """
    bin_files = sorted(glob.glob(os.path.join(frames_dir, "window_*.bin")))
    if not bin_files:
        raise FileNotFoundError(f"No window_*.bin files in {frames_dir}")

    T_total = seq_len + 1
    file_size = os.path.getsize(bin_files[0])
    pix = file_size // (T_total * 3)
    H = W = int(pix ** 0.5)
    if H * W * 3 * T_total != file_size:
        raise ValueError(
            f"Cannot resolve square resolution from file_size={file_size}, "
            f"T+1={T_total}, C=3. Got {H}×{W} which gives {H*W*3*T_total} bytes."
        )
    print(f"[bin] {len(bin_files)} windows, inferred H=W={H}")

    if n_windows is not None:
        bin_files = bin_files[:n_windows]

    batch_wins, batch_starts = [], []
    for path in bin_files:
        # parse start from filename: window_0001234.bin → 1234
        fname = os.path.basename(path)
        try:
            start_idx = int(fname.replace("window_", "").replace(".bin", ""))
        except ValueError:
            start_idx = len(batch_starts)

        raw = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
        win_nhwc = raw.reshape(T_total, H, W, 3)
        win_tchw = np.transpose(win_nhwc, (0, 3, 1, 2))   # [T+1, 3, H, W]
        batch_wins.append(win_tchw)
        batch_starts.append(start_idx)
        if len(batch_wins) == batch_size:
            yield batch_starts, np.stack(batch_wins, axis=0)
            batch_wins, batch_starts = [], []
    if batch_wins:
        yield batch_starts, np.stack(batch_wins, axis=0)


def _iter_images(frames_dir: str, seq_len: int, batch_size: int,
                 frame_stride: int, n_windows=None):
    """
    Sorted PNG/JPG folder → sliding windows. Yields (starts, windows) in order.
    """
    try:
        import cv2
    except ImportError:
        raise SystemExit("opencv-python not installed. Run: pip install opencv-python")

    exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    all_imgs = sorted([f for f in os.listdir(frames_dir) if f.endswith(exts)])
    N = len(all_imgs)
    T_total = seq_len + 1
    S = max(1, int(frame_stride))
    max_start = N - (T_total - 1) * S
    if max_start <= 0:
        raise ValueError(
            f"Not enough frames: need {T_total} with stride {S}, have {N}"
        )

    starts_all = list(range(max_start))
    if n_windows is not None:
        starts_all = starts_all[:n_windows]

    _cache: dict[int, np.ndarray] = {}

    def _load(idx):
        if idx not in _cache:
            path = os.path.join(frames_dir, all_imgs[idx])
            bgr = cv2.imread(path)
            if bgr is None:
                raise IOError(f"Cannot read: {path}")
            _cache[idx] = np.transpose(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (2, 0, 1))
        return _cache[idx]

    batch_wins, batch_starts = [], []
    for s in starts_all:
        idx_list = [s + t * S for t in range(T_total)]
        win = np.stack([_load(i) for i in idx_list], axis=0)   # [T+1, 3, H, W]
        batch_wins.append(win)
        batch_starts.append(s)
        if len(batch_wins) == batch_size:
            yield batch_starts, np.stack(batch_wins, axis=0)
            batch_wins, batch_starts = [], []
        # Evict frames that will not be needed again (keep stride-window of recent frames)
        if len(_cache) > T_total * S + batch_size * T_total:
            oldest = min(_cache.keys())
            del _cache[oldest]
    if batch_wins:
        yield batch_starts, np.stack(batch_wins, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    mg = ap.add_mutually_exclusive_group(required=True)
    mg.add_argument("--weights", metavar="PATH",
                    help="PyTorch .pth state-dict file")
    mg.add_argument("--onnx",    metavar="PATH",
                    help="Exported ONNX model (expects [T,3,H,W] → [3,H,W])")

    dg = ap.add_mutually_exclusive_group(required=True)
    dg.add_argument("--video",      metavar="PATH",
                    help="Input MP4 (must already be resized to --size × --size)")
    dg.add_argument("--frames-dir", metavar="PATH", dest="frames_dir",
                    help="Folder with window_*.bin files (phone capture) "
                         "or sorted PNG/JPG images")

    ap.add_argument("--out",          metavar="PATH", default="psnr_eval.csv",
                    help="Output CSV path (default: psnr_eval.csv)")
    ap.add_argument("--fp16",         action="store_true",
                    help="FP16 inference [PyTorch only]. "
                         "Mimics Android NNAPI USE_FP16 precision.")
    ap.add_argument("--seq-len",      type=int, default=4, metavar="T",
                    help="Number of input frames T (default: 4)")
    ap.add_argument("--size",         type=int, default=224,
                    help="Spatial size H=W in pixels (default: 224, video only)")
    ap.add_argument("--batch-size",   type=int, default=16,
                    help="Windows per inference batch (default: 16)")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="Frame stride for video/image inputs (default: 1)")
    ap.add_argument("--n-windows",    type=int, default=None,
                    help="Cap total windows evaluated (default: all)")
    ap.add_argument("--device",       default="cpu",
                    help="PyTorch device: cpu or cuda (default: cpu)")
    ap.add_argument("--verbose",      action="store_true",
                    help="Print per-batch statistics")
    args = ap.parse_args()

    if args.fp16 and args.onnx:
        print("[warn] --fp16 ignored for ONNX (the .onnx file is FP32; "
              "quantization happens at runtime inside Android NNAPI).")

    # ---- Load model ----
    use_onnx = args.onnx is not None
    if use_onnx:
        sess, in_name, out_name = _load_onnx(args.onnx)
        print(f"[model] ONNX: {args.onnx}")
    else:
        model = _load_pytorch(args.weights, args.seq_len, args.fp16, args.device)
        prec = "fp16" if args.fp16 else "fp32"
        print(f"[model] PyTorch {prec} on {args.device}: {args.weights}")

    # ---- Data iterator ----
    if args.video:
        data_it = _iter_video(
            args.video, seq_len=args.seq_len, batch_size=args.batch_size,
            size=args.size, frame_stride=args.frame_stride,
            n_windows=args.n_windows)
    elif glob.glob(os.path.join(args.frames_dir, "window_*.bin")):
        data_it = _iter_bin(
            args.frames_dir, seq_len=args.seq_len, batch_size=args.batch_size,
            n_windows=args.n_windows)
    else:
        data_it = _iter_images(
            args.frames_dir, seq_len=args.seq_len, batch_size=args.batch_size,
            frame_stride=args.frame_stride, n_windows=args.n_windows)

    # ---- Run inference ----
    rows: list[tuple[int, float]] = []
    t0 = time.time()

    for batch_starts, windows in data_it:
        # windows: [B, T+1, 3, H, W] uint8
        B = windows.shape[0]
        inp_all = windows[:, :-1]   # [B, T, 3, H, W]
        tgt_all = windows[:, -1]    # [B, 3, H, W]

        batch_psnr = []
        for i in range(B):
            inp_u8 = inp_all[i]   # [T, 3, H, W] uint8
            tgt_u8 = tgt_all[i]   # [3, H, W] uint8
            tgt_01 = tgt_u8.astype(np.float32) / 255.0

            if use_onnx:
                pred_norm = _infer_onnx_window(sess, in_name, out_name, inp_u8)
            else:
                pred_norm = _infer_pytorch_window(model, inp_u8, args.fp16, args.device)

            pred_01 = _pred_to_01(pred_norm)
            psnr_db = _psnr_luma_np(pred_01, tgt_01)
            batch_psnr.append(psnr_db)
            rows.append((int(batch_starts[i]), psnr_db))

        if args.verbose:
            m = sum(batch_psnr) / len(batch_psnr)
            print(f"  windows={len(rows)}  batch_mean_psnr={m:.3f} dB")

    elapsed = time.time() - t0

    # ---- Write CSV ----
    rows.sort(key=lambda r: r[0])
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_start", "psnr_db"])
        w.writerows(rows)

    if rows:
        psnr_arr = np.array([p for _, p in rows])
        print(f"\n[done] {len(rows)} windows in {elapsed:.1f}s  "
              f"mean={psnr_arr.mean():.3f} dB  "
              f"min={psnr_arr.min():.3f}  max={psnr_arr.max():.3f}  "
              f"→ {args.out}")
    else:
        print(f"\n[done] 0 windows processed.")


if __name__ == "__main__":
    main()
