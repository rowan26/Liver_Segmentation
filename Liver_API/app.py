# app.py

import shutil
import uuid
import tempfile          # ← ajout : gère /tmp/ sur Linux ET Windows
import nibabel as nib
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from Core.predictor import model, DEVICE, predict

app = FastAPI(
    title="Liver Segmentation API",
    description="3D U-Net — liver & tumor segmentation from NIfTI CT scans",
    version="1.0.0"
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model": "UNet 3D — Run 4 (Dice moyen: 0.6911)"
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):

    # --- 1. Validation du fichier ---
    if not file.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(
            status_code=400,
            detail="Format invalide — envoie un fichier .nii ou .nii.gz"
        )

    # --- 2. Dossier temporaire compatible Windows et Linux ---
    tmp_dir  = tempfile.gettempdir()
    tmp_path = f"{tmp_dir}/{uuid.uuid4()}_{file.filename}"
    out_path = f"{tmp_dir}/{uuid.uuid4()}_mask.nii.gz"

    # --- 3. Sauvegarde du fichier reçu ---
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # --- 4. Inférence ---
    result = predict(tmp_path)
    mask   = result["mask"]
    tumor  = result["tumor_detected"]

    # Guard de sécurité — s'assure que mask est un array numpy
    if isinstance(mask, list):
        mask = np.array(mask[0])
    # --- 5. Sauvegarde du masque prédit ---
    nib.save(
        nib.Nifti1Image(mask.astype(np.uint8), affine=np.eye(4)),
        out_path
    )

    # --- 6. Retourne le masque + détection tumeur dans le header ---
    return FileResponse(
        path=out_path,
        media_type="application/gzip",
        filename="segmentation_mask.nii.gz",
        headers={"X-Tumor-Detected": str(tumor)}
    )