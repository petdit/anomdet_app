from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uuid import uuid4
from datetime import datetime
from typing import Optional
from google.cloud import storage
from infer_light import infer_video
from trainer_light import train_on_video

app = FastAPI()

class TrainRequest(BaseModel):
    gcs_video_uri: str
    device_id: str                      # identifies the phone/area
    out_dir_gcs: Optional[str] = None   # optional override of version folder
    seq_len: int = 4
    epochs: int = 1

    # back-compat synonyms
    augment_mode: str = "none"          # "none" | "lightshift" | "shadow" | "lightshadow"
    loss_mode: str = "mse"              # "mse" | "combined"
    augment: Optional[str] = None
    loss: Optional[str] = None

    alpha: float = 0.7
    beta: float = 0.17
    gamma: float = 0.13

class InferRequest(BaseModel):
    # If use_latest is True, we ignore model_uri/metadata_uri and pull from latest/
    use_latest: bool = False
    device_id: Optional[str] = None

    # Otherwise you can pass explicit URIs
    model_uri: Optional[str] = None
    metadata_uri: Optional[str] = None

    gcs_video_uri: str
    out_dir_gcs: Optional[str] = None
    seq_len: Optional[int] = None
    stride: int = 2
    batch_size: int = 8

def _split_gcs(gcs_uri: str):
    assert gcs_uri.startswith("gs://"), f"Bad GCS URI: {gcs_uri}"
    rest = gcs_uri[5:]
    bucket, path = rest.split("/", 1)
    return bucket, path

def _bucket_from_uri(gcs_uri: str) -> str:
    assert gcs_uri.startswith("gs://")
    rest = gcs_uri[5:]
    bucket, _ = rest.split("/", 1)
    return bucket

def _copy_gcs(src_uri: str, dst_uri: str):
    """
    Server-side copy in GCS: use Bucket.copy_blob (works across buckets too).
    """
    def split(uri):
        rest = uri[5:]
        b, p = rest.split("/", 1)
        return b, p

    sb, sp = split(src_uri)
    db, dp = split(dst_uri)
    client = storage.Client()
    src_bucket = client.bucket(sb)
    dst_bucket = client.bucket(db)
    src_blob = src_bucket.blob(sp)
    # Copy from source bucket → destination bucket with a new name
    src_bucket.copy_blob(src_blob, dst_bucket, dp)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train")
def train(req: TrainRequest):
    try:
        # Back-compat: accept both old and new arg names
        augment_mode = getattr(req, "augment", None) or getattr(req, "augment_mode", None) or "none"
        loss_mode    = getattr(req, "loss", None)    or getattr(req, "loss_mode", None)    or "mse"

        # Decide versioned output folder
        if getattr(req, "out_dir_gcs", None):
            version_dir = req.out_dir_gcs.rstrip("/")
        else:
            bucket = _bucket_from_uri(req.gcs_video_uri)
            stamp  = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
            short  = str(uuid4())[:6]
            device_id = getattr(req, "device_id", "unknown_device")
            version_dir = f"gs://{bucket}/anomaly/devices/{device_id}/versions/{stamp}_{short}"

        # Train (returns model_ts_uri, metadata_uri, weights_uri)
        res = train_on_video(
            gcs_video_uri=req.gcs_video_uri,
            out_dir_gcs=version_dir,
            seq_len=req.seq_len,
            epochs=req.epochs,
            augment_mode=augment_mode,
            loss_mode=loss_mode,
            alpha=req.alpha,
            beta=req.beta,
            gamma=req.gamma,
        )

        # Publish stable latest/ (server+metadata only; phone .pte is produced by exporter)
        device_base = "/".join(version_dir.split("/")[:-2])  # .../anomaly/devices/<device_id>
        latest_dir  = f"{device_base}/latest"
        _copy_gcs(res["model_ts_uri"],  f"{latest_dir}/model.ts.pt")
        _copy_gcs(res["metadata_uri"],  f"{latest_dir}/metadata.json")
        # NOTE: weights.pth stays versioned; exporter reads from <version_dir>.

        return {
            **res,
            "version_uri":          version_dir,
            "latest_model_ts_uri":  f"{latest_dir}/model.ts.pt",
            "latest_metadata_uri":  f"{latest_dir}/metadata.json",
        }

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise

@app.post("/infer")
def infer(req: InferRequest):
    try:
        # Decide output folder
        if req.out_dir_gcs:
            out_dir = req.out_dir_gcs.rstrip("/")
        else:
            bucket = _bucket_from_uri(req.gcs_video_uri)
            stamp  = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
            short  = str(uuid4())[:6]
            device_id = (req.device_id or "unknown_device")
            out_dir = f"gs://{bucket}/anomaly/devices/{device_id}/inference/{stamp}_{short}"

        # Resolve model + metadata
        if req.use_latest:
            if not req.device_id:
                raise HTTPException(status_code=400, detail="use_latest=True requires device_id.")
            bucket = _bucket_from_uri(req.gcs_video_uri)
            latest = f"gs://{bucket}/anomaly/devices/{req.device_id}/latest"
            model_uri = f"{latest}/model.ts.pt"
            metadata_uri = f"{latest}/metadata.json"
        else:
            if not req.model_uri or not req.metadata_uri:
                raise HTTPException(status_code=400, detail="Provide model_uri and metadata_uri or set use_latest=True with device_id.")
            model_uri = req.model_uri
            metadata_uri = req.metadata_uri

        # Run streaming inference (TorchScript)
        res = infer_video(
            model_gcs_uri=model_uri,
            metadata_gcs_uri=metadata_uri,
            gcs_video_uri=req.gcs_video_uri,
            out_dir_gcs=out_dir,
            seq_len=req.seq_len,
            stride=req.stride,
            batch_size=req.batch_size,
        )

        # Publish latest_infer/ if we used the device path
        if not req.out_dir_gcs:
            device_base = "/".join(out_dir.split("/")[:-2])  # .../anomaly/devices/<device_id>
            latest_dir  = f"{device_base}/latest_infer"
            _copy_gcs(res["scores_uri"],  f"{latest_dir}/scores.json")
            _copy_gcs(res["summary_uri"], f"{latest_dir}/summary.json")
            res.update({
                "latest_scores_uri":  f"{latest_dir}/scores.json",
                "latest_summary_uri": f"{latest_dir}/summary.json",
            })

        res["out_dir_gcs"] = out_dir
        return res

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise
