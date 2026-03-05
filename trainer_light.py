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
import subprocess, shlex


IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
IMN_MEAN_5D = IM_MEAN.view(1,1,3,1,1)  # for [B,T,3,H,W]
IMN_STD_5D  = IM_STD.view(1,1,3,1,1)
IMN_MEAN_4D = IM_MEAN.view(1,3,1,1)    # for [B,3,H,W]
IMN_STD_4D  = IM_STD.view(1,3,1,1)



def _get_native_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return float(fps) if fps and fps > 0 else 30.0

#------------ encoder layers helpers ---------


def enable_encoder_bn_affine_only(model):
    """
    Freeze ALL encoder params, then re-enable ONLY BN affine (γ/β) in encoder.
    Also freezes BN running stats by forcing encoder BN modules into eval().
    Call this once after model creation; and call `freeze_encoder_bn_stats(model)` each epoch.
    """
    # freeze everything in the encoder
    for p in model.encoder.parameters():
        p.requires_grad = False

    # unfreeze BN affine and freeze BN stats
    for m in model.encoder.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()  # freeze running mean/var
            if m.affine:
                m.weight.requires_grad = True
                m.bias.requires_grad = True

def freeze_encoder_bn_stats(model):
    """Keep encoder BN in eval() so running stats stay frozen even after model.train()."""
    for m in model.encoder.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def unfreeze_encoder_layer4(model):
    """Make all params in encoder.layer4 trainable (convs + BN affine)."""
    for p in model.encoder.layer4.parameters():
        p.requires_grad = True




# ----------- printing/troubleshooting stuff -------------
def _stats(x: torch.Tensor, label: str):
    # x is a tensor on whatever device
    m, M = float(x.min().item()), float(x.max().item())
    mu, sd = float(x.mean().item()), float(x.std().item())
    print(f"[stats] {label}: min={m:.4f} max={M:.4f} mean={mu:.4f} std={sd:.4f}")

    
def _to_bgr_u8_from_tchw(tchw_u8: np.ndarray) -> np.ndarray:
    """[3,H,W] uint8 RGB -> [H,W,3] uint8 BGR for cv2.imwrite."""
    return cv2.cvtColor(np.transpose(tchw_u8, (1,2,0)), cv2.COLOR_RGB2BGR)

def _save_sequence_jpegs(seq_tchw_u8: np.ndarray, out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    tiles = []
    for t in range(seq_tchw_u8.shape[0]):
        bgr = _to_bgr_u8_from_tchw(seq_tchw_u8[t])
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_{t:02d}.jpg"),
                    bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        tiles.append(bgr)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_strip.jpg"), cv2.hconcat(tiles))




def default_cache_root():
    # User can override in the container / SageMaker job
    return os.environ.get("FRAMECACHE_DIR", "/opt/ml/.cache/frames")


def run_ffmpeg_to_raw(src_video: str, vf: str, threads: int, raw_path: str):
    """
    Runs ffmpeg with the given filter chain and writes raw RGB frames (rgb24)
    to a single binary file at raw_path. This is stream-only: no temp images.
    """
    
    threads_flag = f"-threads {threads}" if threads and threads > 0 else ""
    cmd = (
        f'ffmpeg -hide_banner -loglevel error -nostdin '
        f'{threads_flag} '
        f'-i "{src_video}" '
        f'-an -sn -dn '
        f'-vf "{vf}" '
        f'-f rawvideo -pix_fmt rgb24 -vsync vfr -y -'
    )
    with open(raw_path, "wb") as f:
        # stream stdout (raw bytes) directly to the file
        proc = subprocess.Popen(shlex.split(cmd), stdout=f)
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg exited with code {ret}")
        
def finalize_memmap(raw_path: str, dst_dir: str, size: int,
                    src_video: str):

    bytes_per_frame = size * size * 3
    total_bytes = os.path.getsize(raw_path)
    if total_bytes % bytes_per_frame != 0:
        raise ValueError(f"Raw size {total_bytes} not divisible by frame size {bytes_per_frame}")
    N = total_bytes // bytes_per_frame

    meta = {
        "src_video": src_video,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "shape": {"N": int(N), "H": size, "W": size, "C": 3, "layout": "NHWC", "dtype": "uint8"},
        "filters": {
            "pix_fmt": "rgb24"
        },
        "notes": "raw rgb24 frames written sequentially; use np.memmap with shape [N,H,W,3]"
    }
    with open(os.path.join(dst_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return int(N)


def open_memmap_cache(cache_dir: str):
    """
    Returns (memmap_array, meta_dict), where memmap is np.memmap shaped [N,H,W,3], dtype=uint8.
    """

    meta = json.load(open(os.path.join(cache_dir, "metadata.json"), "r"))
    N, H, W, C = meta["shape"]["N"], meta["shape"]["H"], meta["shape"]["W"], meta["shape"]["C"]
    raw_path = os.path.join(cache_dir, "frames.rgb")
    arr = np.memmap(raw_path, mode="r", dtype=np.uint8, shape=(N, H, W, C))
    return arr, meta

def window_indices(N: int, T: int):
    """
    Sliding windows of length T with stride 1 (i.e., for 5-frame model input+target use T=5).
    Returns list of (start, end) pairs where end is exclusive.
    """
    return [(i, i+T) for i in range(0, max(0, N - T + 1))]



#-------------- read memmap -----------

def extract_to_memmap(src_video: str, cache_root: str,
                      *, size: int = 224,
                      threads: int = 1,
                      overwrite: bool = False) -> tuple[str, int]:



    base = os.path.splitext(os.path.basename(src_video))[0]
    tag = f"size{size}__precrop"

    cache_dir = os.path.join(cache_root, f"{base}__{tag}")
    os.makedirs(cache_dir, exist_ok=True)

    raw_path = os.path.join(cache_dir, "frames.rgb")


    if os.path.exists(raw_path) and not overwrite:
        meta_path = os.path.join(cache_dir, "metadata.json")
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path, "r"))
            H = int(meta["shape"]["H"])
            W = int(meta["shape"]["W"])
            if H != size or W != size:
                raise ValueError(
                    f"Cache exists but shape is {W}x{H}, expected {size}x{size}. "
                    f"Delete cache_dir={cache_dir} or set overwrite_cache=true."
                )
        else:
            # Fallback: check the source video itself
            cap = cv2.VideoCapture(src_video)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            if w != size or h != size:
                raise ValueError(
                    f"Expected pre-cropped video of {size}x{size}, got {w}x{h} for {src_video}. "
                    f"Cache has no metadata.json, refusing to assume geometry."
                )

        N = finalize_memmap(raw_path, cache_dir, size, src_video)
        return cache_dir, N

    vf = "format=rgb24,setsar=1"
    run_ffmpeg_to_raw(src_video, vf, threads, raw_path)

    cap = cv2.VideoCapture(src_video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if w != size or h != size:
        raise ValueError(
            f"Expected pre-cropped video of {size}x{size}, got {w}x{h} for {src_video}. "
            f"Refusing to proceed because AWS-side resizing/cropping is disabled."
        )


    N = finalize_memmap(raw_path, cache_dir, size, src_video)
    return cache_dir, N

def memmap_window_batch_iter(cache_dir: str, *,
                             T: int, batch_size: int,
                             frame_stride: int = 1,
                             sequences_per_epoch: int | None = None,
                             rng=None,
                             return_indices: bool = False):

    """
    Yields:
      - if return_indices=False: torch.uint8 [B,T,3,H,W]
      - if return_indices=True:  (torch.uint8 [B,T,3,H,W], np.ndarray [B] of starts)
    """
    import random as _rnd, numpy as np, torch
    arr, meta = open_memmap_cache(cache_dir)      # arr: [N,H,W,3] uint8
    N, H, W, _ = arr.shape

    S = int(frame_stride) if frame_stride and frame_stride > 0 else 1
    max_start_exclusive = N - (T - 1) * S
    starts = list(range(0, max(0, max_start_exclusive))) 


    rng = rng or _rnd
    rng.shuffle(starts)
    if sequences_per_epoch is not None:
        starts = starts[:sequences_per_epoch]

    batch, batch_idx = [], []
    for s in starts:
        idx = np.arange(s, s + T * S, S, dtype=np.int64)  # length T
        win = arr[idx]                       # [T,H,W,3]
        win = np.transpose(win, (0, 3, 1, 2))     # [T,3,H,W]
        batch.append(win); batch_idx.append(s)
        if len(batch) == batch_size:
            bt = torch.from_numpy(np.stack(batch, axis=0))  # [B,T,3,H,W]
            if return_indices:
                yield bt, np.array(batch_idx, dtype=np.int64)
            else:
                yield bt
            batch, batch_idx = [], []
    if batch:
        bt = torch.from_numpy(np.stack(batch, axis=0))
        if return_indices:
            yield bt, np.array(batch_idx, dtype=np.int64)
        else:
            yield bt




# ------------- frame-folder data loader (backdoor for phone frames + R&D PNG folders) -------------

def frames_folder_batch_iter(frames_dir: str, *,
                             T: int,
                             batch_size: int,
                             frame_stride: int = 1,
                             sequences_per_epoch=None,
                             rng=None,
                             data_format: str = "auto"):
    """
    Yields torch.uint8 batches [B, T, 3, H, W] from pre-extracted frames.

    data_format="bin"   : window_XXXXXXX.bin files saved by Android frame capture.
                          Each file = T+1 frames × H×W×3 bytes (NHWC uint8).
                          frame_stride is embedded in the window files; pass 1 here.
    data_format="image" : sorted PNG/JPG files in frames_dir, builds sliding windows
                          of T+1 frames with the given frame_stride.
    data_format="auto"  : detect by checking if .bin files are present.

    Normalization is NOT applied here — same as memmap_window_batch_iter.
    The training loop handles normalize -> augment -> split.
    """
    import random as _rnd, glob as _glob, os

    frames_dir = str(frames_dir)
    rng = rng or _rnd

    if data_format == "auto":
        has_bin = bool(_glob.glob(os.path.join(frames_dir, "window_*.bin")))
        data_format = "bin" if has_bin else "image"

    if data_format == "bin":
        # --- Phone .bin windows ---
        bin_files = sorted(_glob.glob(os.path.join(frames_dir, "window_*.bin")))
        if not bin_files:
            raise FileNotFoundError(f"No window_*.bin files found in {frames_dir}")

        # Infer H, W from first file: file_size = (T+1) * H * W * 3
        file_size = os.path.getsize(bin_files[0])
        frames_per_file = T + 1
        pix = file_size // (frames_per_file * 3)
        H = W = int(pix ** 0.5)
        if H * W * 3 * frames_per_file != file_size:
            raise ValueError(
                f"Cannot infer square resolution from file size {file_size} "
                f"with T+1={frames_per_file} frames and 3 channels"
            )

        all_files = list(bin_files)
        rng.shuffle(all_files)
        if sequences_per_epoch is not None:
            all_files = all_files[:sequences_per_epoch]

        batch = []
        for path in all_files:
            raw = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
            # shape: [(T+1), H, W, 3]  NHWC
            win_nhwc = raw.reshape(frames_per_file, H, W, 3)
            win_tchw = np.transpose(win_nhwc, (0, 3, 1, 2))   # [T+1, 3, H, W]
            batch.append(win_tchw)
            if len(batch) == batch_size:
                yield torch.from_numpy(np.stack(batch, axis=0))   # [B, T+1, 3, H, W]
                batch = []
        if batch:
            yield torch.from_numpy(np.stack(batch, axis=0))

    elif data_format == "image":
        # --- PNG/JPG folder: sorted files, sliding windows ---
        import cv2 as _cv2
        exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
        all_imgs = sorted([
            f for f in os.listdir(frames_dir)
            if any(f.endswith(e) for e in exts)
        ])
        N = len(all_imgs)
        S = max(1, int(frame_stride))
        frames_needed = T + 1
        max_start = N - (frames_needed - 1) * S
        if max_start <= 0:
            raise ValueError(
                f"Not enough frames: need {frames_needed} with stride {S}, have {N}"
            )

        starts = list(range(max_start))
        rng.shuffle(starts)
        if sequences_per_epoch is not None:
            starts = starts[:sequences_per_epoch]

        # Cache loaded images to avoid re-reading the same file many times
        _cache = {}
        def _load(idx):
            if idx not in _cache:
                path = os.path.join(frames_dir, all_imgs[idx])
                bgr = _cv2.imread(path)
                if bgr is None:
                    raise IOError(f"Failed to read image: {path}")
                rgb = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)
                _cache[idx] = np.transpose(rgb, (2, 0, 1))  # [3, H, W]
            return _cache[idx]

        batch = []
        for s in starts:
            idx_list = [s + t * S for t in range(frames_needed)]
            win = np.stack([_load(i) for i in idx_list], axis=0)  # [T+1, 3, H, W]
            batch.append(win)
            if len(batch) == batch_size:
                yield torch.from_numpy(np.stack(batch, axis=0))
                batch = []
        if batch:
            yield torch.from_numpy(np.stack(batch, axis=0))
    else:
        raise ValueError(f"Unknown data_format: {data_format!r}. Use 'bin', 'image', or 'auto'.")


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

def combined_loss(predicted, target, alpha=0.6, beta=0.25, gamma=0.15):
    p_lin, t_lin = denorm_linear(predicted), denorm_linear(target)
    pix_loss = charbonnier(p_lin - t_lin)
    p01, t01 = denorm01(predicted), denorm01(target)
    grad_loss = gradient_loss_luma(p01, t01)
    ms_ssim_loss = 1 - ms_ssim(luma(p01), luma(t01), data_range=1.0)
    return (alpha * pix_loss + beta * grad_loss + gamma * ms_ssim_loss)

# ----------- augmentation ----------
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
                         base_b_range=(0.9, 1.1),
                         base_c_range=(0.9, 1.1),
                         jitter_amp=0.01,
                         consistent_across_T=False):  # False = per-frame jitter enabled (matches R&D)
    T, C, H, W = seq_tchw.shape
    device, dtype = seq_tchw.device, seq_tchw.dtype

    import random as _rnd
    base_b = float(_rnd.uniform(*base_b_range))
    base_c = float(_rnd.uniform(*base_c_range))

    if consistent_across_T:
        j_b = torch.ones(T,1,1,1, device=device, dtype=dtype)
        j_c = torch.ones(T,1,1,1, device=device, dtype=dtype)
    else:
        j_b = 1.0 + jitter_amp*(2*torch.rand(T,1,1,1, device=device, dtype=dtype)-1.0)
        j_c = 1.0 + jitter_amp*(2*torch.rand(T,1,1,1, device=device, dtype=dtype)-1.0)

    b = base_b * j_b
    c = base_c * j_c

    x = seq_tchw
    # mean-centered contrast (torchvision-style) using grayscale mean per-frame
    Y = 0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3]
    m = Y.mean(dim=(2,3), keepdim=True)                        # [T,1,1,1]
    x = (x - m) * c + m                                        # contrast
    x = (x * b).clamp_(0,1)                                    # brightness
    return x

def _shadow_mask(HW):
    H,W = HW
    mask = np.ones((H,W), dtype=np.float32)
    k = random.randint(3,8)
    poly = np.array([[random.randint(0,W-1), random.randint(0,H-1)] for _ in range(k)], np.int32)
    overlay = np.zeros_like(mask)
    cv2.fillPoly(overlay, [poly], 1.0)
    overlay = cv2.GaussianBlur(overlay, (0,0), sigmaX=random.uniform(5,15))
    alpha = random.uniform(0.3, 0.6)
    shadow = 1.0 - alpha * overlay  # <- darken inside polygon (fix)
    return torch.from_numpy(shadow).float()

def apply_augment(batch_btchw, mode="std"):
    if mode == "none":
        return batch_btchw
    B,T,_,H,W = batch_btchw.shape
    x = batch_btchw.clone()
    import random as _rnd
    # presets:
    # - "std" (default): small affine w.p. 0.7, lightshift 0.5, shadow w.p. 0.2
    # - "general": only affine (always)
    # - "light": only lightshift (always)
    # - "shadow": only shadow (always)
    for b in range(B):
        # decide which effects for this sequence
        use_affine = False; 
        use_light = False; 
        use_shadow = False
        if mode == "std":
            use_affine = (_rnd.random() < 0.7) # ändra te 070502
            use_light  = (_rnd.random() < 0.5)
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
            x[b]  = _lightshift_sequence(x[b], base_b_range=(0.9,1.1), base_c_range=(0.9,1.1), jitter_amp=0.01)
        if use_shadow:
            m = _shadow_mask((H,W)).to(x.device).unsqueeze(0)  # [1,H,W]
            x[b] = (x[b] * m).clamp(0,1)
    return x






def train_on_video_local(local_video_path: str,
                         out_dir_local: str,
                         frame_stride: int = 1,
                         seq_len=4, epochs=1, lr=5e-4,
                         augment_mode="none",
                         loss_mode="combined",
                         alpha=0.6, beta=0.25, gamma=0.15,
                         batch_size=16, size=224,
                         seed=None,
                         save_weights=True, save_ts=True,
                         sequences_per_epoch=4000,
                         vis_sequences: int = 6,
                         threads: int = 0,
                         overwrite_cache: bool = False,
                         data_mode: str = "video"):
    """
    Build/open memmap -> iterate 5-frame windows -> augment full window -> split 4+1 ->
    normalize -> train -> log stats -> save previews/pred-vs-target once/epoch.
    """
    # ---- seeding (optional) ----
    if seed is not None:
        import random as _rnd, numpy as _np
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        _rnd.seed(seed); _np.random.seed(seed)

    os.makedirs(out_dir_local, exist_ok=True)

    # ---- data_mode routing ----
    # "video"  (default): MP4 -> ffmpeg -> memmap cache
    # "frames": local_video_path is a directory of window_*.bin or PNG/JPG frames
    if data_mode == "frames":
        cache_dir = None  # unused in frames mode
        N = None
        print(f"[data] frames mode: dir={local_video_path}")
    else:
        # ---- Build/open cache (SageMaker-friendly) ----
        cache_root = default_cache_root()
        cache_dir, N = extract_to_memmap(
            src_video=local_video_path,
            cache_root=cache_root,
            size=size,
            threads=threads,
            overwrite=overwrite_cache
        )
        print(f"[cache] dir={cache_dir}  frames(N)={N}  windows(T={seq_len+1})={max(0, N-(seq_len+1)+1)}")

    # ---- device / model / optim / loss ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN_TSM(sequence_length=seq_len).to(device)


    # (optional) freeze encoder

    # Full unfreeze (all encoder params trainable)
    for p in model.encoder.parameters():
        p.requires_grad = True

    # Make sure BN layers update running stats during training
    model.train()
    for m in model.encoder.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()   # important: do NOT freeze BN stats

    l4_params   = [p for n, p in model.named_parameters()
                if p.requires_grad and n.startswith("encoder.layer4")]
    rest_params = [p for n, p in model.named_parameters()
                if p.requires_grad and not n.startswith("encoder.layer4")]

    print(f"[opt] rest_params={sum(p.numel() for p in rest_params):,}  "
        f"layer4_params={sum(p.numel() for p in l4_params):,}  "
        f"lr_head={lr}")
    
    opt = optim.Adam(model.parameters(), lr=lr)
    #-------------- encoder stuff end----------------

    #opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr) #opt when encoder frozen. 

    if loss_mode == "combined":
        def loss_fn(pred, tgt):
            return combined_loss(pred, tgt, alpha=alpha, beta=beta, gamma=gamma)
    else:
        loss_fn = nn.MSELoss()


    T = seq_len + 1  # 4-in + 1-target
    saved = 0

    frame_stride = int(frame_stride) if frame_stride is not None else 1
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")

    #sanity check, remove -----------
    def _count_params(pp): return sum(p.numel() for p in pp)
    
    bn_aff = [m.weight for m in model.encoder.modules() if isinstance(m, nn.BatchNorm2d) and m.affine] + \
            [m.bias   for m in model.encoder.modules() if isinstance(m, nn.BatchNorm2d) and m.affine]
    print(f"[BN-affine baseline] trainable params total={_count_params([p for p in model.parameters() if p.requires_grad])} "
        f"(encoder BN-affine={_count_params(bn_aff)})")
    
    mean5 = IMN_MEAN_5D.to(device); std5 = IMN_STD_5D.to(device)
    mean4 = IMN_MEAN_4D.to(device); std4 = IMN_STD_4D.to(device)


    for ep in range(epochs):
        print(f"\n=== epoch {ep+1}/{epochs} ===")

        model.train()

        vis_epoch_dir = os.path.join(out_dir_local, "vis", f"epoch_{ep+1:03d}")
        os.makedirs(vis_epoch_dir, exist_ok=True)
        pred_saved = False  # save pred_vs_target once per epoch

        running_loss = 0.0
        running_count = 0
        first_stats_print = True

        if data_mode == "frames":
            # frames_folder_batch_iter doesn't support return_indices; wrap with enumerate
            _raw_it = frames_folder_batch_iter(
                local_video_path,
                T=seq_len,  # yields [B, T+1, 3, H, W]; the +1 is handled inside
                batch_size=batch_size,
                frame_stride=frame_stride,
                sequences_per_epoch=sequences_per_epoch,
                rng=random,
            )
            # Produce fake start indices for logging compatibility (0, batch_size, 2*batch_size, ...)
            it = (
                (b_u8, np.arange(bi * batch_size, bi * batch_size + b_u8.shape[0], dtype=np.int64))
                for bi, b_u8 in enumerate(_raw_it)
            )
        else:
            it = memmap_window_batch_iter(cache_dir,
                                          T=T, batch_size=batch_size,
                                          frame_stride=frame_stride,
                                          sequences_per_epoch=sequences_per_epoch,
                                          rng=random, return_indices=True)

        for batch_idx, (b_u8, starts) in enumerate(it, start=1):
            # print indices for every sequence in this batch
            if first_stats_print:
                for s in starts.tolist():
                    print(f"  win start={s:06d} -> target={s + seq_len*frame_stride:06d} (stride={frame_stride})")

            # ---- H2D + to [0,1] ----
            if device == "cuda":
                b_u8 = b_u8.pin_memory()
            b = b_u8.to(dtype=torch.float32, device=device, non_blocking=(device=="cuda")).div_(255.0)

            # domain ref (pre-aug)
            if first_stats_print:
                _stats(b, "pre_aug [0,1] (reference)")

            # ---- augment the WHOLE sequence [B,T,3,H,W] identically ----
            b_aug = apply_augment(b, mode=augment_mode)
            b_in  = b_aug[:, :seq_len]   # [B,4,3,H,W]
            tgt01 = b_aug[:,  seq_len]   # [B,3,H,W]

            # domain after aug (actual tensors used downstream)
            if first_stats_print:
                _stats(b_in,  "inputs.post_aug [0,1] (used -> then normalized)")
                _stats(tgt01, "target.post_aug [0,1] (used -> then normalized)")

            # ---- normalize ----
            b_norm   = (b_in  - mean5) / std5
            tgt_norm = (tgt01 - mean4) / std4

            # ---- forward/backward/update ----
            opt.zero_grad(set_to_none=True)
            y = model(b_norm)                       # [B,3,H,W] normalized
            loss = loss_fn(y, tgt_norm)
            loss.backward()
            opt.step()






            # track loss
            bs = int(b.shape[0])
            running_loss += float(loss.item()) * bs
            running_count += bs

            # once per epoch: extra domain checks for predictions
            if first_stats_print:
                _stats(y,           "pred.norm")
                _stats(denorm01(y), "pred.denorm01 [0,1]")

            # ---- save pred vs target once per epoch (first batch is fine) ----
            if not pred_saved:
                with torch.no_grad():
                    pred01 = denorm01(y[0].unsqueeze(0))[0].clamp(0,1).detach().cpu()  # [3,H,W]
                    tgt01v = tgt01[0].clamp(0,1).detach().cpu()
                    pred_u8 = (pred01.mul(255).to(torch.uint8).numpy())
                    tgt_u8  = (tgt01v.mul(255).to(torch.uint8).numpy())
                    side = cv2.hconcat([
                        _to_bgr_u8_from_tchw(pred_u8),
                        _to_bgr_u8_from_tchw(tgt_u8)
                    ])
                    cv2.imwrite(os.path.join(vis_epoch_dir, "pred_vs_target.jpg"), side)
                pred_saved = True

            # numeric guard
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected: {loss.item()}")

            # domain prints for normalized inputs/target (once)
            if first_stats_print:
                _stats(b_norm, "inputs.norm")
                _stats(tgt_norm, "target.norm")
                first_stats_print = False

            # ---- save a few RAW vs AUG sequences for visual inspection ----
            if saved < vis_sequences:
                vis_dir = os.path.join(out_dir_local, "vis", f"epoch_{ep+1:03d}", f"batch_{batch_idx:04d}")
                os.makedirs(vis_dir, exist_ok=True)

                take = min(int(b.shape[0]), vis_sequences - saved)

                raw_np = (b[:take].clamp(0,1).mul(255).to(torch.uint8).cpu().numpy())      # [take,T,3,H,W]
                aug_np = (b_aug[:take].clamp(0,1).mul(255).to(torch.uint8).cpu().numpy())  # [take,T,3,H,W]
                starts_np = starts[:take]

                for k in range(take):
                    sidx = int(starts_np[k])
                    seq_dir = os.path.join(vis_dir, f"seq_start_{sidx:06d}")
                    _save_sequence_jpegs(raw_np[k], seq_dir, prefix="raw")
                    _save_sequence_jpegs(aug_np[k], seq_dir, prefix="aug")
                    saved += 1
                    if saved >= vis_sequences:
                        break

        avg_loss = running_loss / max(1, running_count)
        print(f"[epoch {ep+1}] avg_loss={avg_loss:.6f}  windows={running_count}")

    # ---- export artifacts (honor flags) ----
    if save_weights or save_ts:
        os.makedirs(out_dir_local, exist_ok=True)

    if save_weights:
        torch.save(model.state_dict(), os.path.join(out_dir_local, "weights.pth"))

    if save_ts:
        model_cpu = model.to("cpu").eval()
        ex = torch.randn(1, seq_len, 3, size, size)
        traced = torch.jit.trace(model_cpu, ex, strict=False)
        traced.save(os.path.join(out_dir_local, "model.ts.pt"))
        model.to(device).train()  # restore state


    # --- METADATA (match old schema; add memmap extras safely) ---
    meta = {
        "sequence_length": seq_len,
        "input_size": [3, size, size],
        "train_clip_local": local_video_path,
        "augment": augment_mode,
        "loss_mode": loss_mode,
        "alpha": alpha, "beta": beta, "gamma": gamma,
        "epochs": epochs, "lr": lr,
        "sampler": {
            "native_fps": (_get_native_fps(local_video_path) if data_mode != "frames" else None),
             "frame_stride": frame_stride,
            "seq_len": seq_len,
            "sequences_per_epoch": sequences_per_epoch
        },
        "seed": seed,

        # extras (won’t break old readers; useful for debugging):
        "data_mode": data_mode,
        "framecache": {
            "cache_dir": cache_dir,
            "frames_N": int(N) if N is not None else None,
            "window_T": int(seq_len + 1),
            "scale": size,
            "pix_fmt": "rgb24"
        },
        "encoder_train_strategy": {
            "mode": "full_finetune",
            "encoder_frozen": False,
            "bn_affine_trainable": True,
            "bn_running_stats": "train",
            "layer4_special_handling": False
        }
    }
    local_meta = os.path.join(out_dir_local, "metadata.json")
    with open(local_meta, "w") as f:
        json.dump(meta, f, indent=2)
    return {
        "ok": True,
        "model_ts_path":  os.path.join(out_dir_local, "model.ts.pt") if save_ts else None,
        "metadata_path":  local_meta,
        "weights_path":   os.path.join(out_dir_local, "weights.pth") if save_weights else None
    }