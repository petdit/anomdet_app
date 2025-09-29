import os, json
from paths import SM_TRAIN_IN as SM_TRAIN_DIR, SM_MODEL_DIR
from trainer_light import train_on_video_local

HP_PATH = "/opt/ml/input/config/hyperparameters.json"

def _load_hps():
    if os.path.exists(HP_PATH):
        with open(HP_PATH) as f:
            raw = json.load(f)  # values are strings
        def coerce(v):
            try: return json.loads(v)  # allows "null","1","0.5","true"
            except Exception: return v
        return {k: coerce(v) for k, v in raw.items()}
    return {}

def _env(name, cast, default):
    v = os.getenv(name)
    if v is None: return default
    try: return cast(v)
    except Exception: return default

def _env_bool(name, default):
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip().lower() in {"1","true","yes","y"}

def _find_single_mp4(search_dir: str) -> str:
    for root, _, files in os.walk(search_dir):
        for f in files:
            if f.lower().endswith(".mp4"):
                return os.path.join(root, f)
    raise RuntimeError(f"No .mp4 found under {search_dir}. Did SageMaker mount the input?")


def main():
    video = _find_single_mp4(SM_TRAIN_DIR)
    h = _load_hps()



    

    epochs     = int(h.get("epochs",     _env("ET_EPOCHS",     int,   1)))
    lr         = float(h.get("lr",       _env("ET_LR",         float, 5e-4)))
    augment    = str(h.get("augment",    _env("ET_AUGMENT",    str,   "std")))
    loss_mode  = str(h.get("loss_mode",  _env("ET_LOSS_MODE",  str,   "combined")))
    alpha      = float(h.get("alpha",    _env("ET_ALPHA",      float, 0.7)))
    beta       = float(h.get("beta",     _env("ET_BETA",       float, 0.17)))
    gamma      = float(h.get("gamma",    _env("ET_GAMMA",      float, 0.13)))
    seq_len    = int(h.get("seq_len",    _env("ET_SEQ_LEN",    int,   4)))
    batch_size = int(h.get("batch_size", _env("ET_BATCH_SIZE", int,   16)))
    size       = int(h.get("size",       _env("ET_SIZE",       int,   224)))
    seed       = h.get("seed",           _env("ET_SEED",       int,   None))
    keep_every = h.get("keep_every",     _env("ET_KEEP_EVERY", int, None))
    lr_encoder_l4 = float(h.get("lr_encoder_l4", _env("ET_LR_ENCODER_L4", float, 5e-5)))
    threads       = int(h.get("threads",       _env("ET_THREADS",       int,   0)))   # 0 = ffmpeg auto
    overwrite     = bool(h.get("overwrite_cache", _env_bool("ET_OVERWRITE_CACHE", False)))
    vis_sequences = int(h.get("vis_sequences", _env("ET_VIS_SEQUENCES",  int,   0)))
    sequences_per_epoch = h.get("sequences_per_epoch",  _env("ET_SEQUENCES_PER_EPOCH", int,   None))
    
    # optional training artifact toggles (default True)
    save_weights = bool(h.get("save_weights", _env_bool("ET_SAVE_WEIGHTS", True)))
    save_ts      = bool(h.get("save_ts",      _env_bool("ET_SAVE_TS",      True)))
    roi_val = h.get("roi", None)
    if isinstance(roi_val, str):
        try:
            import json as _json
            roi_val = tuple(_json.loads(roi_val))
        except Exception:
            parts = [p.strip() for p in roi_val.split(",")]
            roi_val = tuple(int(x) for x in parts) if len(parts) == 4 else None
    elif isinstance(roi_val, list):
        roi_val = tuple(roi_val)
    elif not isinstance(roi_val, tuple):
        roi_val = None

    if keep_every is not None and keep_every < 2:
        keep_every = None  # treat 0/1 as "keep all"

    res = train_on_video_local(
        video, SM_MODEL_DIR,
        seq_len=seq_len, epochs=epochs, lr=lr,
        augment_mode=augment, loss_mode=loss_mode,
        alpha=alpha, beta=beta, gamma=gamma,
        batch_size=batch_size, size=size,
        seed=seed, save_weights=save_weights, save_ts=save_ts,
        keep_every=keep_every, sequences_per_epoch=sequences_per_epoch, 
        roi=roi_val, threads=threads, overwrite_cache=overwrite,
        lr_encoder_l4=lr_encoder_l4, vis_sequences=vis_sequences
    )
    print(json.dumps({"ok": True, "resolved": {
        "epochs": epochs, "lr": lr, "seq_len": seq_len,
        "augment": augment, "loss_mode": loss_mode,
        "alpha": alpha, "beta": beta, "gamma": gamma,
        "batch_size": batch_size, "size": size,
        "keep_every": keep_every, "sequences_per_epoch": sequences_per_epoch, "roi": roi_val, 
        "seed": seed, "save_weights": save_weights, "save_ts": save_ts
    }, "result": res}))

if __name__ == "__main__":
    main()
