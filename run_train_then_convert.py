import os, sys, json, argparse, shutil, subprocess


from paths import SM_TRAIN_IN, SM_MODEL_DIR, SM_PROC_IN, SM_PROC_OUT



def run(cmd, env=None):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd, env=env)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="Optional path to an MP4. If omitted, one must exist under /opt/ml/input/data/training/")
    ap.add_argument("--backend", default=os.getenv("ET_BACKEND", "vulkan"),
                    choices=["vulkan", "xnnpack", "nnapi"])
    ap.add_argument("--skip-convert", action="store_true", help="Run training only.")

    args = ap.parse_args()

    os.makedirs(SM_TRAIN_IN, exist_ok=True)
    os.makedirs(SM_MODEL_DIR, exist_ok=True)
    os.makedirs(SM_PROC_IN, exist_ok=True)
    os.makedirs(SM_PROC_OUT, exist_ok=True)

    # stage input for training (support either --input or a pre-mounted file)
    staged = os.path.join(SM_TRAIN_IN, "video.mp4")

    def _find_mp4(root: str):
        for r, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith(".mp4"):
                    return os.path.join(r, f)
        return None

    if args.input:
        if not (os.path.isfile(args.input) and args.input.lower().endswith(".mp4")):
            sys.exit("ERROR: --input must be a readable .mp4 file")
        if os.path.abspath(args.input) != os.path.abspath(staged):
            shutil.copy2(args.input, staged)
    else:
        existing = _find_mp4(SM_TRAIN_IN)
        if not existing:
            sys.exit(
                "ERROR: No .mp4 found under /opt/ml/input/data/training.\n"
                "Either pass --input /path/to/video.mp4, or mount a file to "
                "/opt/ml/input/data/training/video.mp4.\n"
                'Example (PowerShell): -v "${PWD}\\video.mp4:/opt/ml/input/data/training/video.mp4:ro"'
            )
        if os.path.abspath(existing) != os.path.abspath(staged):
            shutil.copy2(existing, staged)

    # 1) TRAIN
    env = dict(os.environ)                # 1) take the parent process env
    if os.getenv("ET_SEED") not in (None, ""):
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # 2) add one key
    run(["/opt/train-venv/bin/python", "train_entry.py"], env=env)  # 3) launch child

    def _env_bool(name, default=False):
        v = os.getenv(name)
        if v is None: return default
        return str(v).strip().lower() in {"1","true","yes","y"}

    skip_convert = args.skip_convert or _env_bool("ET_SKIP_CONVERT", False)

    if skip_convert:
        print(json.dumps({"ok": True, "phase": "train_only", "model_src": SM_MODEL_DIR}))
        return

    # Clean processing input to avoid stale files
    for f in os.listdir(SM_PROC_IN):
        p = os.path.join(SM_PROC_IN, f)
        try:
            if os.path.isfile(p): os.remove(p)
        except Exception:
            pass

    # Copy model artifacts to converter's input
    tarball = os.path.join(SM_MODEL_DIR, "model.tar.gz")
    if os.path.isfile(tarball):
        shutil.copy2(tarball, os.path.join(SM_PROC_IN, "model.tar.gz"))
    else:
        for fname in os.listdir(SM_MODEL_DIR):
            src = os.path.join(SM_MODEL_DIR, fname)
            dst = os.path.join(SM_PROC_IN, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

    # 2) CONVERT
    conv_cmd = ["/opt/convert-venv/bin/python", "convert_entry.py", "--backend", args.backend]
    run(conv_cmd)
    print(json.dumps({"ok": True, "phase": "full",
                    "model_src": SM_MODEL_DIR, "convert_out": SM_PROC_OUT}))

if __name__ == "__main__":
    main()
