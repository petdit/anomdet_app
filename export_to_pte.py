# export_to_pte.py
import os, json, tempfile, argparse, torch
from google.cloud import storage
from torch.export import export as torch_export
from executorch.exir import to_edge_transform_and_lower

# Your model
from model_min import CNN_TSM

def _split_gcs(uri: str):
    assert uri.startswith("gs://"), f"Bad GCS URI: {uri}"
    rest = uri[5:]
    b, p = rest.split("/", 1)
    return b, p

def gcs_download(uri: str, local_path: str):
    b, p = _split_gcs(uri)
    storage.Client().bucket(b).blob(p).download_to_filename(local_path)

def gcs_upload(local_path: str, uri: str):
    b, p = _split_gcs(uri)
    storage.Client().bucket(b).blob(p).upload_from_filename(local_path)

def _get_partitioners(backend: str):
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        return [XnnpackPartitioner()]
    elif backend == "vulkan":
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
        return [VulkanPartitioner()]
    elif backend == "nnapi":
        # Commonly not available via the pip wheel; prefer xnnpack or vulkan.
        raise RuntimeError("NNAPI partitioner not available in this build. Use --backend xnnpack or --backend vulkan.")
    else:
        raise RuntimeError(f"Unknown backend: {backend}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version_dir", required=True,
                    help="gs://.../anomaly/devices/<device_id>/versions/<stamp_id>")
    ap.add_argument("--backend", default="vulkan", choices=["nnapi", "vulkan", "xnnpack"],
                    help="ExecuTorch backend to lower to.")
    args = ap.parse_args()

    version_dir = args.version_dir.rstrip("/")
    weights_uri = f"{version_dir}/weights.pth"
    meta_uri    = f"{version_dir}/metadata.json"
    out_pte_uri = f"{version_dir}/model.{args.backend}.pte"

    # Also publish to latest/ so the phone has a stable pointer
    device_base = "/".join(version_dir.split("/")[:-2])  # .../anomaly/devices/<device_id>
    latest_dir  = f"{device_base}/latest"

    with tempfile.TemporaryDirectory() as td:
        # Download artifacts
        local_w = os.path.join(td, "weights.pth")
        local_m = os.path.join(td, "metadata.json")
        gcs_download(weights_uri, local_w)
        gcs_download(meta_uri,    local_m)

        meta = json.load(open(local_m, "r"))
        seq_len = int(meta.get("sequence_length", 4))

        # Rebuild model and load weights
        model = CNN_TSM(sequence_length=seq_len).eval()
        sd = torch.load(local_w, map_location="cpu")
        model.load_state_dict(sd)

        # torch.export → ExportedProgram (shape-specialized)
        example = torch.randn(1, seq_len, 3, 224, 224)  # match runtime contract
        ep = torch_export(model, (example,))

        # ExecuTorch: Lower with the chosen partitioner, then serialize .pte
        partitioners = _get_partitioners(args.backend)   # list of partitioners
        lowered = to_edge_transform_and_lower(ep, partitioner=partitioners)
        et_prog = lowered.to_executorch()

        local_pte = os.path.join(td, f"model.{args.backend}.pte")
        # In 0.7, et_prog exposes a bytes buffer
        with open(local_pte, "wb") as f:
            f.write(et_prog.buffer)

        # Upload to the version folder and publish to latest/
        gcs_upload(local_pte, out_pte_uri)
        gcs_upload(local_pte, f"{latest_dir}/model.pte")

        print(json.dumps({
            "ok": True,
            "version_dir": version_dir,
            "backend": args.backend,
            "pte_uri": out_pte_uri,
            "latest_pte_uri": f"{latest_dir}/model.pte"
        }))

if __name__ == "__main__":
    main()
