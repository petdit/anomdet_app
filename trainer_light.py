import os, json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model_min import CNN_TSM
import random
from pytorch_msssim import ms_ssim
from collections import deque
import time


IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
IMN_MEAN_5D = IM_MEAN.view(1,1,3,1,1)  # for [B,T,3,H,W]
IMN_STD_5D  = IM_STD.view(1,1,3,1,1)
IMN_MEAN_4D = IM_MEAN.view(1,3,1,1)    # for [B,3,H,W]
IMN_STD_4D  = IM_STD.view(1,3,1,1)

# --- augmentation helpers: sequence-wise affine & lightshift ---
def _affine_sequence(seq_tchw: torch.Tensor, angle_deg: float, tx_frac: float, ty_frac: float) -> torch.Tensor:
    """
    seq_tchw: [T,3,H,W] in [0,1]. Applies the same small affine to all frames in the sequence.
    angle_deg: rotation in degrees; tx/ty are fractions of width/height (e.g., 0.02 = 2%).
    """
    T, C, H, W = seq_tchw.shape
    theta = torch.zeros((1, 2, 3), device=seq_tchw.device, dtype=seq_tchw.dtype)
    ang = torch.tensor(angle_deg * np.pi / 180.0, device=seq_tchw.device, dtype=seq_tchw.dtype)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta[0,0,0] =  cos; theta[0,0,1] = -sin; theta[0,0,2] = 2.0 * tx_frac
    theta[0,1,0] =  sin; theta[0,1,1] =  cos; theta[0,1,2] = 2.0 * ty_frac
    grid = F.affine_grid(theta.expand(T, -1, -1), size=(T, C, H, W), align_corners=False)
    return F.grid_sample(seq_tchw, grid, mode="bilinear", padding_mode="border", align_corners=False).clamp(0,1)

def _lightshift_sequence(seq_tchw: torch.Tensor,
                         base_b_range=(0.95, 1.05),
                         base_c_range=(0.95, 1.05),
                         jitter_amp=0.03) -> torch.Tensor:
    """
    Sequence-global base brightness/contrast with per-frame jitter (vectorized).
    """
    import random as _rnd
    T, _, _, _ = seq_tchw.shape
    base_b = float(_rnd.uniform(*base_b_range))
    base_c = float(_rnd.uniform(*base_c_range))
    # independent jitter for brightness & contrast per frame
    j_b = 1.0 + jitter_amp * (2.0 * torch.rand(T, 1, 1, 1, device=seq_tchw.device, dtype=seq_tchw.dtype) - 1.0)
    j_c = 1.0 + jitter_amp * (2.0 * torch.rand(T, 1, 1, 1, device=seq_tchw.device, dtype=seq_tchw.dtype) - 1.0)
    bfac = base_b * j_b
    cfac = base_c * j_c
    out = (seq_tchw * cfac).clamp_(0, 1)
    out.mul_(bfac).clamp_(0, 1)
    return out


def _get_native_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return float(fps) if fps and fps > 0 else 30.0

def _keep_every_for_fps(video_path: str, fps: float | None) -> int:
    if fps is None or fps <= 0:
        return 1
    native = _get_native_fps(video_path)
    return max(1, int(round(native / float(fps))))


def frames_iter(video_path: str, size=224, keep_every=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    idx = 0
    while True:
        ok = cap.grab()  # decode but do not allocate/convert yet
        if not ok:
            break
        if (idx % keep_every) == 0:
            ok, frame_bgr = cap.retrieve()
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
            yield square
        idx += 1
    cap.release()

def _estimate_num_windows(video_path: str, T=4, keep_every=1) -> int:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if total <= 0:
        # slow fallback
        cap = cv2.VideoCapture(video_path)
        total = 0
        while cap.grab():
            total += 1
        cap.release()
    kept = (total + keep_every - 1) // keep_every
    return max(0, kept - T + 1)


def prob_window_batch_iter(video_path: str, *, T=4, batch_size=4,
                           size=224, keep_every=1,
                           sequences_per_epoch: int | None = None, rng=None):
    import random as _rnd
    rng = rng or _rnd
    # compute acceptance probability p so E[kept] ~= frames_per_epoch
    if sequences_per_epoch  is None:
        p = 1.0
    else:
        N_est = _estimate_num_windows(video_path, T=T, keep_every=keep_every)
        p = 1.0 if N_est <= 0 else min(1.0, float(sequences_per_epoch) / float(N_est))

    buf = deque(maxlen=T)
    batch = []
    for f in frames_iter(video_path, size=size, keep_every=keep_every):
        buf.append(f)
        if len(buf) < T:
            continue
        if rng.random() < p:  # Bernoulli accept
            win = np.stack(list(buf), axis=0)  # [T,H,W,3] #moved float32 conv to gpu. uint8 now.
            win = np.transpose(win, (0,3,1,2))                             # [T,3,H,W] uint8
            batch.append(win)
            if len(batch) == batch_size:
                yield torch.from_numpy(np.stack(batch, axis=0))            # [B,T,3,H,W]
                batch = []
        # slide by 1 kept frame (full overlap)
        buf.popleft()

    if batch:
        yield torch.from_numpy(np.stack(batch, axis=0))





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

def apply_augment(batch_btchw, mode="std"):
    if mode == "none":
        return batch_btchw
    B,T,_,H,W = batch_btchw.shape
    x = batch_btchw.clone()
    import random as _rnd
    # presets:
    # - "std" (default): small affine w.p. 0.7, lightshift always, shadow w.p. 0.2
    # - "general": only affine (always)
    # - "light": only lightshift (always)
    # - "shadow": only shadow (always)
    for b in range(B):
        # decide which effects for this sequence
        use_affine = False; use_light = False; use_shadow = False
        if mode == "std":
            use_affine = (_rnd.random() < 0.7)
            use_light  = True
            use_shadow = (_rnd.random() < 0.2)
        elif mode == "general":
            use_affine = True
        elif mode == "light":
            use_light = True
        elif mode == "shadow":
            use_shadow = True

        # apply in sequence: affine -> lightshift -> shadow
        if use_affine:
            angle = _rnd.uniform(-3.0, 3.0)
            tx    = _rnd.uniform(-0.02, 0.02)
            ty    = _rnd.uniform(-0.02, 0.02)
            x[b]  = _affine_sequence(x[b], angle_deg=angle, tx_frac=tx, ty_frac=ty)
        if use_light:
            x[b]  = _lightshift_sequence(x[b], base_b_range=(0.95,1.05), base_c_range=(0.95,1.05), jitter_amp=0.03)
        if use_shadow:
            m = _shadow_mask((H,W)).to(x.device).unsqueeze(0)  # [1,H,W]
            x[b] = (x[b] * m).clamp(0,1)
    return x


def train_on_video_local(local_video_path: str,
                         out_dir_local: str,
                         seq_len=4, epochs=1, lr=5e-4,
                         augment_mode="none",
                         loss_mode="combined",
                         alpha=0.7, beta=0.17, gamma=0.13,
                         batch_size=16, size=224,
                         seed=None,
                         save_weights=True, save_ts=True, 
                         fps=None, sequences_per_epoch=None):
    if seed is not None:
        import random, numpy as np
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        random.seed(seed); np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    # timing toggle (env only; no CLI/HP needed)
    TIMING = str(os.getenv("ET_TIMING", "0")).strip().lower() in {"1","true","yes","y"}
    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

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
    model.encoder.eval()
    keep_every = _keep_every_for_fps(local_video_path, fps)

    for ep in range(epochs):
        epoch_start = time.perf_counter()
        running = 0.0   
        num_wins = 0
        batches  = 0
        data_sec = h2d_sec = aug_sec = norm_sec = fwd_sec = loss_sec = bwd_sec = opt_sec = 0.0
        t_prev = time.perf_counter()  # time until first batch arrives

        for b in prob_window_batch_iter(local_video_path,
                                        T=seq_len, batch_size=batch_size,
                                        size=size, keep_every=keep_every,
                                        sequences_per_epoch=sequences_per_epoch):
            if TIMING:
                # time spent waiting for next batch (decode/resize/sampling)
                now = time.perf_counter()
                data_sec += (now - t_prev)

            if TIMING: t_h2d0 = time.perf_counter()
            b = b.pin_memory()                       # pin host memory for faster H2D
            b = b.to(device, non_blocking=True)      # async H2D copy
            if TIMING: _sync(); t_h2d1 = time.perf_counter(); h2d_sec += (t_h2d1 - t_h2d0)
            b = b.float().div_(255.0)

            if TIMING: _sync(); t0 = time.perf_counter()
            b = apply_augment(b, mode=augment_mode)
            if TIMING:
                _sync(); t1 = time.perf_counter(); aug_sec += (t1 - t0)
            b_norm   = (b - mean5) / std5
            if TIMING:
                _sync(); t2 = time.perf_counter(); norm_sec += (t2 - t1)

            tgt_norm = (b[:, -1] - mean4) / std4   
            opt.zero_grad()
            if TIMING: _sync(); t3 = time.perf_counter()
            y = model(b_norm)
            if TIMING:
                _sync(); t4 = time.perf_counter(); fwd_sec += (t4 - t3)

            if TIMING: _sync(); t5 = time.perf_counter()
            loss = loss_fn(y, tgt_norm)
            if TIMING:
                _sync(); t5b = time.perf_counter(); loss_sec += (t5b - t5)
            loss.backward()
            if TIMING:
                _sync(); t6 = time.perf_counter(); bwd_sec += (t6 - t5b)
            opt.step()
            if TIMING:
                _sync(); t7 = time.perf_counter(); opt_sec += (t7 - t6); t_prev = time.perf_counter()

            running += float(loss.item()) * b.shape[0]
            num_wins += b.shape[0]
            batches += 1
        avg_loss = running / max(1, num_wins)
        if TIMING:
            total_train = data_sec + h2d_sec + aug_sec + norm_sec + fwd_sec + loss_sec + bwd_sec + opt_sec
            epoch_wall = time.perf_counter() - epoch_start
            other = epoch_wall - total_train
            print(
                f"epoch {ep+1}: windows={num_wins}, batches={batches}, avg_loss={avg_loss:.6f}\n"
                f"  data={data_sec:.3f}s  h2d={h2d_sec:.3f}s  aug={aug_sec:.3f}s  norm={norm_sec:.3f}s  "
                f"fwd={fwd_sec:.3f}s  loss={loss_sec:.3f}s  bwd={bwd_sec:.3f}s  opt={opt_sec:.3f}s  "
                f"total(train)={total_train:.3f}s  other={max(other,0):.3f}s  total(wall)={epoch_wall:.3f}s"
            )
        else:
            print(f"epoch {ep+1}: windows={num_wins}, avg_loss={avg_loss:.6f}")
    # ----- EXPORTS -----

    model.eval()
    os.makedirs(out_dir_local, exist_ok=True)

    meta = {
        "sequence_length": seq_len,
        "input_size": [3, size, size],
        "train_clip_local": local_video_path,
        "augment": augment_mode,
        "loss_mode": loss_mode,
        "alpha": alpha, "beta": beta, "gamma": gamma,
        "epochs": epochs, "lr": lr,
        "sampler": {
            "native_fps": _get_native_fps(local_video_path),
            "fps": fps,                    # requested target fps (None means native)
            "keep_every": keep_every,      # computed from fps
            "seq_len": seq_len,
            "stride_effective": 1,         # we slide by 1 kept frame
            "sequences_per_epoch": sequences_per_epoch
        },
        "seed": seed,
    }
    local_meta = os.path.join(out_dir_local, "metadata.json")
    with open(local_meta, "w") as f:
        json.dump(meta, f)

    cur_device = next(model.parameters()).device
    model_cpu = model.to("cpu").eval()

    if save_ts:
        ex_cpu = torch.randn(1, seq_len, 3, size, size, device="cpu")
        traced = torch.jit.trace(model_cpu, ex_cpu, strict=False)
        local_model_ts = os.path.join(out_dir_local, "model.ts.pt")
        traced.save(local_model_ts)
    else:
        local_model_ts = None

    if save_weights or (not save_ts):  # ensure at least one artifact exists
        local_ckpt = os.path.join(out_dir_local, "weights.pth")
        torch.save(model_cpu.state_dict(), local_ckpt)
    else:
        local_ckpt = None

    model.to(cur_device)

    return {
        "ok": True,
        "model_ts_path":  local_model_ts,
        "metadata_path":  local_meta,
        "weights_path":   local_ckpt
    }
