

# Anomaly Trainer — Dev README

## What this is

A tiny FastAPI service that:

1. downloads a short video from Google Cloud Storage (GCS),
2. trains your CNN+TSM reconstruction model on that “normal” clip,
3. exports a TorchScript **Lite** model (`.ptl`) for Android/Metal performance,
4. uploads artifacts back to GCS in a clean, phone-friendly layout.

---

## Repo layout (service/)

```
service/
  app.py                 # FastAPI app: /health, /train
  trainer_light.py       # video → frames → train → export → upload
  model_min.py           # CNN_TSM + TSMFinal
  Dockerfile.gpu         # GPU image (CUDA runtime + PyTorch + deps)
  requirements.gpu.txt   # Python deps (NumPy 1.26, FastAPI, etc.)
```

---

## Requirements (Windows, Docker, GPU)

* **Docker Desktop** (Windows, using WSL2 backend).
* **NVIDIA GPU drivers** installed (Windows).
* **NVIDIA Container Toolkit** enabled by Docker Desktop (Settings → Resources → GPU: enable).
* **Google Cloud SDK** (for `gsutil` and `gcloud`) on host to manage buckets.
* A **service account** JSON key with at least these bucket-level roles on your data bucket:

  * `roles/storage.objectAdmin`
  * `roles/storage.legacyBucketReader`

Put the key at, e.g., `C:\creds\avvikelser-ae259a4b5b97.json`.

---

## One-time GCS setup (host)

Create/upload a test clip and grant permissions once:

```powershell
# (if not created) make bucket
# gsutil mb -p <PROJECT_ID> -l <REGION> gs://anomaly-proto-data

# upload a test video
gsutil cp "C:\path\to\sample.mp4" gs://anomaly-proto-data/test_clips/sample.mp4

# grant bucket-level roles to the service account
gsutil iam ch serviceAccount:anomaly-trainer-sa@avvikelser.iam.gserviceaccount.com:roles/storage.objectAdmin gs://anomaly-proto-data
gsutil iam ch serviceAccount:anomaly-trainer-sa@avvikelser.iam.gserviceaccount.com:roles/storage.legacyBucketReader gs://anomaly-proto-data
```

---

## Build the GPU image

From `service/`:

```powershell
docker build -t anomaly-trainer:gpu -f Dockerfile.gpu --progress=plain .
```

> First build is large (CUDA + PyTorch). Subsequent rebuilds are much faster.

---

## Run the container (GPU)

```powershell
docker run --name anomaly-trainer-local --gpus all -p 8080:8080 `
  -v "C:\creds\avvikelser-ae259a4b5b97.json:/creds.json:ro" `
  -e GOOGLE_APPLICATION_CREDENTIALS=/creds.json `
  -e GOOGLE_CLOUD_PROJECT=avvikelser `
  anomaly-trainer:gpu
```

* Leaves the server running in the terminal:

  * `GET  /health` → `{"status":"ok"}`
  * `POST /train`  → kicks off train/export/upload

---

## Send a training request

Create `request.json` (PowerShell here-string):

```powershell
@'
{
  "gcs_video_uri": "gs://anomaly-proto-data/test_clips/sample.mp4",
  "device_id": "lineA-station3",
  "seq_len": 4,
  "epochs": 1,
  "augment_mode": "lightshadow",
  "loss_mode": "combined",
  "alpha": 0.7,
  "beta": 0.17,
  "gamma": 0.13
}
'@ | Out-File -FilePath request.json -Encoding ascii
```

Send it:

```powershell
curl.exe -H "Content-Type: application/json" -d "@request.json" http://localhost:8080/train
```

Expected response (example):

```json
{
  "ok": true,
  "model_uri": "gs://anomaly-proto-data/anomaly/devices/lineA-station3/versions/2025-08-20_14-03-27_ab12cd/model.ptl",
  "metadata_uri": "gs://anomaly-proto-data/anomaly/devices/lineA-station3/versions/2025-08-20_14-03-27_ab12cd/metadata.json",
  "version_uri": "gs://anomaly-proto-data/anomaly/devices/lineA-station3/versions/2025-08-20_14-03-27_ab12cd",
  "latest_model_uri": "gs://anomaly-proto-data/anomaly/devices/lineA-station3/latest/model.ptl",
  "latest_metadata_uri": "gs://anomaly-proto-data/anomaly/devices/lineA-station3/latest/metadata.json"
}
```

---

## Where outputs go (phone-friendly layout)

The service writes to **versioned** and **latest** locations:

```
gs://anomaly-proto-data/anomaly/devices/<device_id>/
  latest/
    model.ptl
    metadata.json
  versions/
    YYYY-MM-DD_HH-MM-SS_<shortid>/
      model.ptl
      metadata.json
```

Phones always fetch:

```
.../devices/<device_id>/latest/model.ptl
```

---

## Useful commands

### Check the server

```powershell
curl.exe http://localhost:8080/health
```

### Inspect bucket structure

```powershell
gsutil ls -r gs://anomaly-proto-data/anomaly/devices/**
gsutil ls -r gs://anomaly-proto-data/anomaly/devices/lineA-station3/**
gsutil du -h -r gs://anomaly-proto-data/anomaly/devices/lineA-station3/**
```

### Download artifacts locally

```powershell
gsutil cp gs://anomaly-proto-data/anomaly/devices/lineA-station3/latest/model.ptl .
gsutil cp gs://anomaly-proto-data/anomaly/devices/lineA-station3/latest/metadata.json .
```

### Stop/remove the container

```powershell
docker stop anomaly-trainer-local
docker rm anomaly-trainer-local
```

### Rebuild after code changes

```powershell
docker build -t anomaly-trainer:gpu -f Dockerfile.gpu --progress=plain .
```

### Free up space (careful!)

```powershell
# remove stopped containers & dangling images
docker system prune -f

# aggressive: remove all images + containers (DATA LOSS of images/containers)
# docker system prune -a
```

> Docker on Windows stores data inside the WSL2 virtual disk. Pruning frees *Docker’s* cache, but Windows disk space might not update until WSL compacts its VHD (Docker Desktop → Troubleshoot → Clean / Purge data).

---

## Notes on training details

* Frames are extracted at **native FPS**, cropped to **square** without stretching (top-crop for tall videos, center-crop for wide), then resized to **224×224**.
* Augmentation options (in `trainer_light.py`): `"none" | "lightshift" | "shadow" | "lightshadow"`.
* Loss options: `"mse"` or `"combined"` (Charbonnier on pixel, gradient on luma, MS-SSIM on luma).
* We normalize inputs/targets with ImageNet mean/std, matching the ResNet backbone.

---

## GPU sanity

* Container uses GPU automatically if available:

  ```python
  device = "cuda" if torch.cuda.is_available() else "cpu"
  ```
* You can watch GPU usage in another terminal:

  ```powershell
  nvidia-smi
  ```

---

## Common issues & fixes

* **403 on GCS** during training: ensure the container is started with

  ```
  -v "C:\creds\your-key.json:/creds.json:ro"
  -e GOOGLE_APPLICATION_CREDENTIALS=/creds.json
  ```

  and the **bucket** has:

  * `roles/storage.objectAdmin`
  * `roles/storage.legacyBucketReader`

* **Can list/read object but cannot “list buckets”**: expected if you didn’t grant project-level viewer; not required.

* **MP4 read fails**: make sure `ffmpeg` is installed in the image (it is in `Dockerfile.gpu`), and your video codec is common (H.264/AAC).

* **TorchScript tracing warnings**:

  * “Converting a tensor to a Python boolean” — benign for fixed shapes.
  * “Outputs not close” — tiny float diffs from BN/ConvTranspose; safe.

* **Image size huge**: CUDA + PyTorch is big. To keep size down:

  * Use `-slim` bases where possible.
  * Avoid installing dev tools.
  * Stick to minimal `requirements.gpu.txt` and pin versions (done).

---

## Example: quick re-run

```powershell
# 1) stop old run (if needed)
docker stop anomaly-trainer-local 2>$null
docker rm anomaly-trainer-local 2>$null

# 2) start again
docker run --name anomaly-trainer-local --gpus all -p 8080:8080 `
  -v "C:\creds\avvikelser-ae259a4b5b97.json:/creds.json:ro" `
  -e GOOGLE_APPLICATION_CREDENTIALS=/creds.json `
  -e GOOGLE_CLOUD_PROJECT=avvikelser `
  anomaly-trainer:gpu

# 3) train
curl.exe -H "Content-Type: application/json" -d "@request.json" http://localhost:8080/train
```

---

If you want, I can also add a tiny **GET `/devices/{device_id}/latest`** endpoint for the phone to query the URIs—totally optional since the paths are stable.
