import os, tarfile, json, shutil, tempfile, argparse
import torch
import onnx
import onnxruntime as ort
from model_min import CNN_TSM_NNAPI  

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



def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version to export with (default: 18).",
    )
    ap.add_argument(
        "--onnx-out",
        default="model.onnx",
        help="Output ONNX filename (default: model.onnx).",
    )

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
        help="Export precision: 16 for fp16, 32 for fp32 (JSON/env override this). Have not tested 16 very extensively, may be shit",
    )

    args = ap.parse_args()

    # allow env vars to toggle in SageMaker without changing args
    save_weights = args.save_weights or _env_flag("ET_SAVE_WEIGHTS", False)
    save_ts = args.save_ts or _env_flag("ET_SAVE_TS", False)
    convert_precision = _resolve_convert_precision(args.convert_precision)
    to_half = convert_precision == "16"
    os.makedirs(OUT_DIR, exist_ok=True)
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

        size = 224
        in_size = m.get("input_size", [3, 224, 224]) if meta and os.path.isfile(meta) else [3, 224, 224]
        if isinstance(in_size, (list, tuple)) and len(in_size) == 3:
            size = int(in_size[1])
        



        # Build model + example input (preferred path: plain nn.Module with loaded weights)
        model = None

        example = torch.randn(seq_len, 3, size, size)  # [T,3,S,S]
        if weights and os.path.isfile(weights):
            model = CNN_TSM_NNAPI(sequence_length=seq_len, spatial_size=size).eval()
            sd = torch.load(weights, map_location="cpu")
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"Loaded weights into CNN_TSM_NNAPI. Missing={len(missing)} Unexpected={len(unexpected)}")
            if to_half:
                model = model.half()
                example = example.half()
        elif ts_pt and os.path.isfile(ts_pt):
            # Fallback: ONNX export directly from TorchScript if needed
            model = torch.jit.load(ts_pt, map_location="cpu").eval()
            if to_half:
                model = model.half()
                example = example.half()
        else:
            raise RuntimeError("Could not find weights.pth or model.ts.pt in input artifacts")


        os.makedirs(OUT_DIR, exist_ok=True)
        onnx_path = os.path.join(OUT_DIR, args.onnx_out)

        # Export (static T; keep batch=1 static for mobile perf unless you need dynamic)
        torch.onnx.export(
            model,
            example,
            onnx_path,
            opset_version=args.opset,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
            dynamic_axes=None,  # set to {"input": {0: "batch"}, "output": {0: "batch"}} ONLY if you truly need variable B
            keep_initializers_as_inputs=False,   # <<< important
            export_params=True
        )
        print(f"Wrote {onnx_path}")




        # always mirror metadata next to the ONNX if available
        if meta and os.path.isfile(meta):
            shutil.copy2(meta, os.path.join(OUT_DIR, "metadata.json"))

        # optional: copy raw trained artifacts for desktop testing
        if save_weights and weights and os.path.isfile(weights):
            shutil.copy2(weights, os.path.join(OUT_DIR, "weights.pth"))
            if meta and os.path.isfile(meta):
                shutil.copy2(meta, os.path.join(OUT_DIR, "metadata.json"))

        if save_ts and ts_pt and os.path.isfile(ts_pt):
            shutil.copy2(ts_pt, os.path.join(OUT_DIR, "model.ts.pt"))

        out = {
            "ok": True,
            "format": "onnx",
            "opset": args.opset,
            "convert_precision": convert_precision,
            "outputs": {"onnx": onnx_path}
        }



        # --- Merge into metadata.json if it exists next to the produced ONNX/ORT ---
        def try_merge_out_to_metadata(out_dict, model_path):
            # Look for metadata.json in the same directory as model_path
            d = os.path.dirname(model_path)
            metadata_candidate = os.path.join(d, "metadata.json")
            if os.path.exists(metadata_candidate):
                try:
                    with open(metadata_candidate, "r", encoding="utf-8") as fh:
                        metadata = json.load(fh)
                except Exception as e:
                    print(f"WARNING: failed to read existing metadata.json at {metadata_candidate}: {e}")
                    metadata = {}

                # Decide which keys to merge — avoid overwriting training-only fields unless explicit
                metadata.setdefault("converter", {})
                metadata["converter"].update({
                    "ok": out_dict.get("ok"),
                    "format": out_dict.get("format"),
                    "opset": out_dict.get("opset"),
                    "convert_precision": out_dict.get("convert_precision"),
                    "outputs": out_dict.get("outputs")
                })

                try:
                    with open(metadata_candidate, "w", encoding="utf-8") as fh:
                        json.dump(metadata, fh, indent=2, sort_keys=False)
                    print(f"Merged converter output into {metadata_candidate}")
                except Exception as e:
                    print(f"WARNING: failed to write merged metadata.json at {metadata_candidate}: {e}")
            else:
                print(f"No metadata.json next to model at {d}; skipping merge")

        # call it with the ONNX path (prefer onnx path variable)
        if  onnx_path:
            try_merge_out_to_metadata(out, onnx_path)


    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
