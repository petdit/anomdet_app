# infer_light.py (streaming, TorchScript-on-GPU)
import os, json, tempfile, collections
from typing import List, Deque
import cv2, numpy as np, torch
from google.cloud import storage

# Reuse helpers & constants from trainer_light.py
from trainer_light import (
    _split_gcs, gcs_download, gcs_upload, extract_frames,
    IM_MEAN, IM_STD
)

IM_MEAN_4D = IM_MEAN.view(1,3,1,1)
IM_STD_4D  = IM_STD.view(1,3,1,1)
IM_MEAN_5D = IM_MEAN.view(1,1,3,1,1)
IM_STD_5D  = IM_STD.view(1,1,3,1,1)


def _psnr_y(pred01: torch.Tensor, tgt01: torch.Tensor) -> torch.Tensor:
    """
    PSNR on luma in [0,1]. pred01/tgt01: [B,3,H,W] in [0,1].
    """
    r,g,b = pred01[:,0:1], pred01[:,1:2], pred01[:,2:3]
    Yp = 0.299*r + 0.587*g + 0.114*b
    r,g,b = tgt01[:,0:1], tgt01[:,1:2], tgt01[:,2:3]
    Yt = 0.299*r + 0.587*g + 0.114*b
    mse = torch.mean((Yp - Yt)**2, dim=(1,2,3))
    return 10.0 * torch.log10(1.0 / (mse + 1e-8))


def _iter_windows_stream(frames: List[np.ndarray], T: int, stride: int):
    """
    Yield windows as numpy arrays of shape [T,3,224,224] in [0,1] without storing all windows.
    """
    if len(frames) < T + 1:
        return
    buf: Deque[np.ndarray] = collections.deque(maxlen=T)
    for i, f in enumerate(frames):
        chw = np.transpose((f.astype(np.float32) / 255.0), (2,0,1))  # [3,H,W] in [0,1]
        buf.append(chw)
        # once we have T frames, emit every 'stride', but only if there is a target frame after the window
        if len(buf) == T and (i - (T-1)) % stride == 0 and (i + 1) < len(frames):
            win = np.stack(list(buf), axis=0)  # [T,3,H,W]
            yield win


def load_model_for_infer(model_gcs_uri: str):
    """
    Download model from GCS and load it for server inference.
    - Accepts only TorchScript '.ts.pt' (GPU/CPU capable).
    - Rejects ExecuTorch '.pte' (phone-only artifact).
    Returns: (model, device)
    """
    with tempfile.TemporaryDirectory() as td:
        local_model = os.path.join(td, os.path.basename(model_gcs_uri))
        gcs_download(model_gcs_uri, local_model)

        if local_model.endswith(".ts.pt"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = torch.jit.load(local_model, map_location=device).eval()
            return model, device

        elif local_model.endswith(".pte"):
            raise ValueError(
                "Provided model is an ExecuTorch .pte (phone artifact). "
                "Use the TorchScript '.ts.pt' for server inference."
            )

        else:
            raise ValueError(
                "Unsupported model format. Provide a TorchScript '.ts.pt' file."
            )


def infer_video(
    model_gcs_uri: str,
    metadata_gcs_uri: str,
    gcs_video_uri: str,
    out_dir_gcs: str,
    seq_len: int | None = None,
    stride: int = 1,
    batch_size: int = 8
):
    """
    Streamed inference over a video with constant memory.
    Loads TorchScript (.ts.pt) model on CUDA if available, otherwise CPU.
    Writes scores.json and summary.json to out_dir_gcs.
    """
    # Download metadata & video
    with tempfile.TemporaryDirectory() as td:
        local_meta  = os.path.join(td, "metadata.json")
        local_video = os.path.join(td, "clip.mp4")
        gcs_download(metadata_gcs_uri, local_meta)
        gcs_download(gcs_video_uri, local_video)

        with open(local_meta, "r") as f:
            meta = json.load(f)
        T = int(seq_len or meta.get("sequence_length", 4))

        # Load TorchScript model
        model, device = load_model_for_infer(model_gcs_uri)

        # Frames already use your crop-top-to-square → resize(224) policy
        frames = extract_frames(local_video, size=224)
        if len(frames) < T + 1:
            raise RuntimeError("Not enough frames for inference windows.")

        mean5 = IM_MEAN_5D.to(device); std5 = IM_STD_5D.to(device)
        mean4 = IM_MEAN_4D.to(device); std4 = IM_STD_4D.to(device)

        scores: List[float] = []
        batch_list: List[np.ndarray] = []
        
        processed = 0
        with torch.no_grad():
            for win in _iter_windows_stream(frames, T=T, stride=stride):
                batch_list.append(win)  # win: [T,3,224,224]
                if len(batch_list) == batch_size:
                    x = torch.from_numpy(np.stack(batch_list)).to(device)  # [B,T,3,H,W]
                    x_norm = (x - mean5) / std5
                    y = model(x_norm)                       # [B,3,224,224] normalized domain
                    y01 = (y * std4 + mean4).clamp(0,1)     # back to [0,1]
                    tgt01 = x[:, -1]                        # last frame in [0,1]
                    ps = _psnr_y(y01, tgt01).detach().cpu().tolist()
                    processed += len(ps)
                    if processed % 200 == 0:
                        print(f"[infer] processed {processed} windows…", flush=True)
                    scores.extend(ps)
                    batch_list.clear()
                    if device == "cuda":
                        torch.cuda.empty_cache()
            # flush tail
            if batch_list:
                x = torch.from_numpy(np.stack(batch_list)).to(device)
                x_norm = (x - mean5) / std5
                y = model(x_norm)
                y01 = (y * std4 + mean4).clamp(0,1)
                tgt01 = x[:, -1]
                ps = _psnr_y(y01, tgt01).detach().cpu().tolist()
                scores.extend(ps)

        # Aggregate & upload results
        ps = np.array(scores, dtype=np.float32)
        mu, std = float(ps.mean()), float(ps.std() + 1e-6)
        anomaly = (1.0 - (ps - mu) / std).tolist()

        out_scores  = os.path.join(td, "scores.json")
        out_summary = os.path.join(td, "summary.json")
        with open(out_scores, "w") as f:
            json.dump({"psnr": scores, "anomaly": anomaly}, f)
        with open(out_summary, "w") as f:
            json.dump({"seq_len": T, "stride": stride, "mean_psnr": mu, "std_psnr": std}, f)

        gcs_upload(out_scores,  f"{out_dir_gcs.rstrip('/')}/scores.json")
        gcs_upload(out_summary, f"{out_dir_gcs.rstrip('/')}/summary.json")

        return {
            "ok": True,
            "scores_uri":  f"{out_dir_gcs.rstrip('/')}/scores.json",
            "summary_uri": f"{out_dir_gcs.rstrip('/')}/summary.json",
            "n_windows": len(scores),
            "psnr_mean":  mu,
            "psnr_std":   std
        }
