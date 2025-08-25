import os, json, tempfile
from typing import List
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from google.cloud import storage
from model_min import CNN_TSM
import random
from pytorch_msssim import ms_ssim

IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
IMN_MEAN_5D = IM_MEAN.view(1,1,3,1,1)  # for [B,T,3,H,W]
IMN_STD_5D  = IM_STD.view(1,1,3,1,1)
IMN_MEAN_4D = IM_MEAN.view(1,3,1,1)    # for [B,3,H,W]
IMN_STD_4D  = IM_STD.view(1,3,1,1)

# ---------- GCS helpers ----------
def _split_gcs(gcs_uri: str):
    assert gcs_uri.startswith("gs://")
    rest = gcs_uri[5:]
    bucket, path = rest.split("/", 1)
    return bucket, path

def gcs_download(gcs_uri: str, local_path: str):
    bucket, path = _split_gcs(gcs_uri)
    storage.Client().bucket(bucket).blob(path).download_to_filename(local_path)

def gcs_upload(local_path: str, gcs_uri: str):
    bucket, path = _split_gcs(gcs_uri)
    storage.Client().bucket(bucket).blob(path).upload_from_filename(local_path)

# ---------- video -> frames ----------
def extract_frames(video_path: str, size=224):
    """
    Reads every frame, crops to a square without stretching,
    then resizes to size×size. For tall videos (H>W) we keep the bottom WxW.
    For wide videos (W>H), we center-crop.
    Returns: list of HWC RGB uint8 frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        H, W, _ = frame.shape
        if H == W:
            square = frame
        elif H > W:
            y0 = H - W
            square = frame[y0:H, 0:W]
        else:
            x0 = (W - H) // 2
            square = frame[0:H, x0:x0+H]
        square = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(square)
    cap.release()
    return frames

# ------------- loss functions ------------
def denorm01(x):
    m, s = IM_MEAN.to(x.device), IM_STD.to(x.device)
    return (x * s + m).clamp(0, 1)

def denorm_linear(x):
    m, s = IM_MEAN.to(x.device), IM_STD.to(x.device)
    return (x * s + m)

def luma(x01):
    r,g,b = x01[:,0:1], x01[:,1:2], x01[:,2:3]
    return 0.299*r + 0.587*g + 0.114*b

def charbonnier(residual, eps=1e-6):
    return torch.sqrt(residual*residual + eps).mean()

def gradient_loss_luma(p01, t01):
    Yp, Yt = luma(p01), luma(t01)
    dxp = (Yp[...,1:] - Yp[...,:-1]).abs()
    dyp = (Yp[:,:,1:,:] - Yp[:,:,:-1,:]).abs()
    dxt = (Yt[...,1:] - Yt[...,:-1]).abs()
    dyt = (Yt[:,:,1:,:] - Yt[:,:,:-1,:]).abs()
    return (dxp - dxt).abs().mean() + (dyp - dyt).abs().mean()

def combined_loss(predicted, target, alpha=0.7, beta=0.17, gamma=0.13):
    p_lin, t_lin = denorm_linear(predicted), denorm_linear(target)
    pix_loss = charbonnier(p_lin - t_lin)
    p01, t01 = denorm01(predicted), denorm01(target)
    grad_loss = gradient_loss_luma(p01, t01)
    ms_ssim_loss = 1 - ms_ssim(luma(p01), luma(t01), data_range=1.0)
    return (alpha * pix_loss + beta * grad_loss + gamma * ms_ssim_loss)

# ----------- augmentation ----------
def _brightness_contrast(x, b_range=(0.95,1.05), c_range=(0.95,1.05)):
    b = random.uniform(*b_range)
    c = random.uniform(*c_range)
    x = x * c
    x = x * b
    return x.clamp(0,1)

def _shadow_mask(HW):
    H,W = HW
    mask = np.ones((H,W), dtype=np.float32)
    k = random.randint(3,8)
    poly = np.array([[random.randint(0,W-1), random.randint(0,H-1)] for _ in range(k)], np.int32)
    overlay = np.zeros_like(mask)
    cv2.fillPoly(overlay, [poly], 1.0)
    overlay = cv2.GaussianBlur(overlay, (0,0), sigmaX=random.uniform(5,15))
    alpha = random.uniform(0.3, 0.6)
    shadow = 1.0 - alpha * (1.0 - overlay)
    return torch.from_numpy(shadow).float()  # [H,W]

def apply_augment(batch_btchw, mode="none"):
    """
    batch_btchw: [B,T,3,224,224] in [0,1]
    mode: "none" | "lightshift" | "shadow" | "lightshadow"
    """
    if mode == "none":
        return batch_btchw
    B,T,_,H,W = batch_btchw.shape
    x = batch_btchw
    if "light" in mode:
        for b in range(B):
            for t in range(T):
                x[b,t] = _brightness_contrast(x[b,t])
    if "shadow" in mode:
        for b in range(B):
            m = _shadow_mask((H,W)).to(x.device)  # [H,W]
            m = m.unsqueeze(0)                    # [1,H,W]
            x[b] = (x[b] * m).clamp(0,1)
    return x

def make_windows(frames: List[np.ndarray], T=4, stride=2):
    if len(frames) < T + 1:
        return []
    arr = (np.stack(frames).astype(np.float32) / 255.0)  # [N,H,W,3]
    arr = np.transpose(arr, (0,3,1,2))                   # [N,3,H,W]
    windows = []
    for s in range(0, arr.shape[0] - T + 1, stride):
        windows.append(arr[s:s+T])                       # [T,3,224,224]
    return windows

def batches(windows, bs=4):
    for i in range(0, len(windows), bs):
        yield torch.from_numpy(np.stack(windows[i:i+bs]))  # [B,T,3,224,224]

# ---------- training + export ----------
def train_on_video(gcs_video_uri: str, out_dir_gcs: str,
                   seq_len=4, epochs=1, lr=1e-3,
                   augment_mode="none",
                   loss_mode="mse",
                   alpha=0.7, beta=0.17, gamma=0.13):

    with tempfile.TemporaryDirectory() as td:
        local_video = os.path.join(td, "clip.mp4")
        gcs_download(gcs_video_uri, local_video)

        frames = extract_frames(local_video, size=224)
        wins = make_windows(frames, T=seq_len, stride=2)
        if len(wins) < 4:
            raise RuntimeError("Not enough frames after sampling; try recording longer.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CNN_TSM(sequence_length=seq_len).to(device)
        mean5 = IMN_MEAN_5D.to(device); std5 = IMN_STD_5D.to(device)
        mean4 = IMN_MEAN_4D.to(device); std4 = IMN_STD_4D.to(device)

        # freeze encoder for quick adaptation
        for p in model.encoder.parameters():
            p.requires_grad = False

        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        if loss_mode == "combined":
            def loss_fn(pred, tgt):
                return combined_loss(pred, tgt, alpha=alpha, beta=beta, gamma=gamma)
        else:
            loss_fn = nn.MSELoss()

        model.train()
        for ep in range(epochs):
            running = 0.0
            for b in batches(wins, bs=4):
                b = b.to(device)                         # [B,T,3,224,224] in [0,1]
                b = apply_augment(b, mode=augment_mode)  # augment in [0,1]
                b_norm   = (b - mean5) / std5            # [B,T,3,H,W]
                tgt_norm = (b[:, -1] - mean4) / std4     # [B,3,H,W]
                opt.zero_grad()
                y = model(b_norm)                         # normalized domain
                loss = loss_fn(y, tgt_norm)
                loss.backward()
                opt.step()
                running += float(loss.item())
            print(f"epoch {ep+1}: avg_loss={running / max(1, len(wins)):.6f}")

        # ----- EXPORTS -----
        model.eval()

        # metadata
        meta = {
            "sequence_length": seq_len,
            "input_size": [3, 224, 224],
            "train_clip": gcs_video_uri,
            "augment": augment_mode,
            "loss_mode": loss_mode,
            "alpha": alpha, "beta": beta, "gamma": gamma
        }
        local_meta = os.path.join(td, "metadata.json")
        with open(local_meta, "w") as f:
            json.dump(meta, f)

        # (A) TorchScript (server/GPU)
        cur_device = next(model.parameters()).device
        model_cpu = model.to("cpu").eval()
        ex_cpu = torch.randn(1, seq_len, 3, 224, 224, device="cpu")
        traced = torch.jit.trace(model_cpu, ex_cpu, strict=False)
        local_model_ts = os.path.join(td, "model.ts.pt")
        traced.save(local_model_ts)

        # (B) Save weights for phone exporter (.pte is produced in separate container)
        local_ckpt = os.path.join(td, "weights.pth")
        torch.save(model_cpu.state_dict(), local_ckpt)

        # Upload artifacts
        gcs_upload(local_model_ts,  f"{out_dir_gcs}/model.ts.pt")
        gcs_upload(local_ckpt,      f"{out_dir_gcs}/weights.pth")
        gcs_upload(local_meta,      f"{out_dir_gcs}/metadata.json")

        # (Optional) restore original device
        model.to(cur_device)

        return {
            "ok": True,
            "model_ts_uri":  f"{out_dir_gcs}/model.ts.pt",  # server inference
            "metadata_uri":  f"{out_dir_gcs}/metadata.json",
            "weights_uri":   f"{out_dir_gcs}/weights.pth"
        }
