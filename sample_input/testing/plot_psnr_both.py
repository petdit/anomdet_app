# plot_psnr_both.py
import argparse, csv, os
import numpy as np
import matplotlib.pyplot as plt

def smooth(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1: return y
    k = int(k)
    w = np.ones(k, dtype=np.float64) / k
    return np.convolve(y, w, mode="same")

def main():
    ap = argparse.ArgumentParser(description="Plot RGB-PSNR and Luma-PSNR from metrics.csv")
    ap.add_argument("--csv", required=True, help="Path to metrics.csv (from infer_cli)")
    ap.add_argument("--out", default=None, help="Output PNG path (if omitted, shows interactively)")
    ap.add_argument("--title", default=None, help="Plot title")
    ap.add_argument("--smooth", type=int, default=1, help="Moving-average window (samples)")
    ap.add_argument("--step", type=int, default=1, help="Plot every Nth sample")
    ap.add_argument("--eps", type=float, default=1e-12, help="Numerical epsilon for PSNR calc")
    args = ap.parse_args()

    xs, psnr_rgb, luma_mse = [], [], []
    with open(args.csv, "r", newline="") as f:
        r = csv.DictReader(f)
        # expected columns: start_idx,target_idx,psnr_rgb,luma_mse
        for row in r:
            xs.append(int(row.get("target_idx", row.get("start_idx", len(xs)))))
            psnr_rgb.append(float(row["psnr_rgb"]))
            luma_mse.append(float(row["luma_mse"]))

    x = np.asarray(xs)
    y_rgb = np.asarray(psnr_rgb, dtype=np.float64)
    lmse  = np.asarray(luma_mse, dtype=np.float64)

    # Luma-PSNR from luma-MSE
    lmse = np.clip(lmse, args.eps, None)
    y_luma = -10.0 * np.log10(lmse)

    # Smoothing (same window for both)
    y_rgb  = smooth(y_rgb,  args.smooth)
    y_luma = smooth(y_luma, args.smooth)

    # Optional downsample
    if args.step > 1:
        x      = x[::args.step]
        y_rgb  = y_rgb[::args.step]
        y_luma = y_luma[::args.step]

    # Plot (single axis, both in dB)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    l1, = ax.plot(x, y_rgb,  label="PSNR (RGB)",  color="tab:blue")
    l2, = ax.plot(x, y_luma, label="PSNR (Luma)", color="tab:orange", alpha=0.9)

    ax.set_xlabel("Frame index (target)")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")

    if args.title:
        ax.set_title(args.title)
    else:
        ax.set_title(f"PSNR metrics: {os.path.basename(args.csv)}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"wrote {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
