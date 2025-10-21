import os, tarfile, json, shutil, tempfile, argparse
import torch
from model_min import CNN_TSM
from paths import SM_PROC_IN as IN_DIR, SM_PROC_OUT as OUT_DIR

HP_PATH = "/opt/ml/input/config/hyperparameters.json"


def _load_hps():
    if os.path.exists(HP_PATH):
        try:
            with open(HP_PATH, encoding="utf-8-sig") as f:  # tolerate BOM
                raw = json.load(f)

            def coerce(v):
                try:
                    return json.loads(v)
                except Exception:
                    return v

            return {k: coerce(v) for k, v in raw.items()}
        except Exception:
            pass
    return {}


def _resolve_convert_precision(cli_value: str | None):
    """
    Precedence: hyperparameters.json -> env ET_CONVERT_PRECISION -> CLI value.
    Returns "16" or "32". Default = "32" to match legacy behavior.
    """
    h = _load_hps()
    val = str(
        h.get(
            "convert_precision",
            os.getenv("ET_CONVERT_PRECISION", cli_value if cli_value is not None else "32"),
        )
    ).strip().lower()

    if val in {"16", "fp16", "half"}:
        return "16"
    if val in {"32", "fp32", "float"}:
        return "32"
    print("[warn] Unsupported convert_precision; defaulting to 32", flush=True)
    return "32"


def _find(path, names):
    if isinstance(names, str):
        names = [names]
    for root, _, files in os.walk(path):
        for f in files:
            if f in names:
                return os.path.join(root, f)
    return None


def _get_partitioners(backend: str):
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

        return [XnnpackPartitioner()]
    elif backend == "vulkan":
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

        return [VulkanPartitioner()]
    elif backend == "nnapi":
        raise RuntimeError("NNAPI partitioner not available in this build; use xnnpack or vulkan.")
    else:
        raise RuntimeError(f"Unknown backend: {backend}")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backend",
        default=None,
        choices=["xnnpack", "vulkan", "nnapi"],
        help="ExecuTorch backend to lower to.",
    )
    # simple, optional copies of the trained artifacts to OUT_DIR:
    ap.add_argument(
        "--save-weights",
        action="store_true",
        help="Copy weights.pth (+metadata.json if present) to converter output.",
    )
    ap.add_argument(
        "--save-ts",
        action="store_true",
        help="Copy model.ts.pt to converter output if present.",
    )
    ap.add_argument(
        "--convert-precision",
        choices=["16", "32"],
        default=None,
        help="Export precision: 16 for fp16, 32 for fp32 (JSON/env override this).",
    )

    args = ap.parse_args()

    # allow env vars to toggle in SageMaker without changing args
    save_weights = args.save_weights or _env_flag("SAVE_WEIGHTS", False)
    save_ts = args.save_ts or _env_flag("SAVE_TS", False)
    convert_precision = _resolve_convert_precision(args.convert_precision)
    to_half = convert_precision == "16"
    os.makedirs(OUT_DIR, exist_ok=True)
    backend = args.backend or os.getenv("ET_BACKEND") or "vulkan"
    tmp = tempfile.mkdtemp()
    try:
        # support model.tar.gz (SageMaker) or plain folder
        tar_path = _find(IN_DIR, "model.tar.gz")
        if tar_path:
            with tarfile.open(tar_path, "r:gz") as t:
                t.extractall(tmp)
            search_dir = tmp
        else:
            search_dir = IN_DIR

        weights = _find(search_dir, "weights.pth")
        meta = _find(search_dir, "metadata.json")
        ts_pt = _find(search_dir, "model.ts.pt")  # optional fallback

        seq_len = 4
        if meta and os.path.isfile(meta):
            with open(meta, "r", encoding="utf-8") as f:
                m = json.load(f)
            seq_len = int(m.get("sequence_length", 4))

        # preferred export path: rebuild model + load weights
        if weights and os.path.isfile(weights):
            model = CNN_TSM(sequence_length=seq_len).eval()
            sd = torch.load(weights, map_location="cpu")
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"Loaded weights. Missing={len(missing)} Unexpected={len(unexpected)}")

            from torch.export import export as torch_export

            example = torch.randn(1, seq_len, 3, 224, 224)
            if to_half:
                model = model.half()
                example = example.half()
            ep = torch_export(model, (example,))

        elif ts_pt and os.path.isfile(ts_pt):
            # fallback if only TorchScript exists
            traced = torch.jit.load(ts_pt, map_location="cpu").eval()
            from torch.export import export as torch_export

            example = torch.randn(1, seq_len, 3, 224, 224)
            if to_half:
                traced = traced.half()
                example = example.half()
            ep = torch_export(traced, (example,))
        else:
            raise RuntimeError("Could not find weights.pth or model.ts.pt in input artifacts")

        # lower to ExecuTorch and write .pte
        from executorch.exir import to_edge_transform_and_lower

        lowered = to_edge_transform_and_lower(ep, partitioner=_get_partitioners(backend))
        et_prog = lowered.to_executorch()

        out_pte = os.path.join(OUT_DIR, f"model.{backend}.pte")
        with open(out_pte, "wb") as f:
            f.write(et_prog.buffer)
        print(f"Wrote {out_pte}")

        # always mirror metadata next to the .pte if available
        if meta and os.path.isfile(meta):
            shutil.copy2(meta, os.path.join(OUT_DIR, "metadata.json"))

        # optional: copy raw trained artifacts for desktop testing
        if save_weights and weights and os.path.isfile(weights):
            shutil.copy2(weights, os.path.join(OUT_DIR, "weights.pth"))
            if meta and os.path.isfile(meta):
                shutil.copy2(meta, os.path.join(OUT_DIR, "metadata.json"))

        if save_ts and ts_pt and os.path.isfile(ts_pt):
            shutil.copy2(ts_pt, os.path.join(OUT_DIR, "model.ts.pt"))

        print(
            json.dumps(
                {"ok": True, "output": out_pte, "backend": backend, "convert_precision": convert_precision}
            )
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
