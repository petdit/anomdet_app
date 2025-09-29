# plot_metrics.py
import argparse, csv, os
import numpy as np
import matplotlib.pyplot as plt

def smooth(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1: return y
    k = int(k)
    w = np.ones(k, dtype=np.float64) / k
    return np.convolve(y, w, mode="same")

def main():
    ap = argparse.ArgumentParser(description="Plot PSNR and luma-MSE from metrics.csv")
    ap.add_argument("--csv", required=True, help="Path to metrics.csv")
    ap.add_argument("--out", default=None, help="Output PNG path (if omitted, shows interactively)")
    ap.add_argument("--title", default=None, help="Plot title")
    ap.add_argument("--smooth", type=int, default=1, help="Moving-average window (samples)")
    ap.add_argument("--step", type=int, default=1, help="Plot every Nth sample")
    ap.add_argument("--log-luma", action="store_true", help="Log10 scale for luma-MSE axis")
    args = ap.parse_args()

    xs, psnr, lmse = [], [], []

    with open(args.csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(int(row["target_idx"]))
            psnr.append(float(row["psnr_rgb"]))
            lmse.append(float(row["luma_mse"]))

    x = np.asarray(xs)
    y_psnr = np.asarray(psnr, dtype=np.float64)
    y_lmse = np.asarray(lmse, dtype=np.float64)

    # smoothing (same window/centering for both)
    y_psnr = smooth(y_psnr, args.smooth)
    y_lmse = smooth(y_lmse, args.smooth)

    # optional downsample for heavy traces
    if args.step > 1:
        x = x[::args.step]
        y_psnr = y_psnr[::args.step]
        y_lmse = y_lmse[::args.step]

    fig, ax1 = plt.subplots(figsize=(12, 4.5))

    l1, = ax1.plot(x, y_psnr, label="PSNR (RGB) [dB]")
    ax1.set_xlabel("Frame index (target)")
    ax1.set_ylabel("PSNR (dB)")
    ax1.grid(True, which="both", alpha=0.25)

    ax2 = ax1.twinx()
    l2, = ax2.plot(x, y_lmse, label="Luma MSE", alpha=0.8)
    ax2.set_ylabel("Luma MSE")
    if args.log_luma:
        ax2.set_yscale("log")

    # unified legend
    lines = [l1, l2]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="best")

    if args.title:
        plt.title(args.title)
    else:
        base = os.path.basename(args.csv)
        plt.title(f"Metrics: {base}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"wrote {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
