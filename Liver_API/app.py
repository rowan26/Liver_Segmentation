import shutil
import uuid
import tempfile
import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from Core.predictor import model, DEVICE, predict
from scipy.ndimage import zoom
import os
import matplotlib.pyplot as plt


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

    # --- 1. Validation du format du fichier ---
    if not file.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(
            status_code=400,
            detail="Format invalide — envoie un fichier .nii ou .nii.gz"
        )

    # --- 2. Configuration des chemins temporaires ---
    tmp_dir  = tempfile.gettempdir()
    tmp_path = f"{tmp_dir}/{uuid.uuid4()}_{file.filename}"
    # On enregistre en .nii brut pour faciliter la lecture en mémoire côté Streamlit
    out_path = f"{tmp_dir}/{uuid.uuid4()}_mask.nii"

    # --- 3. Sauvegarde locale du fichier reçu ---
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # --- 4. Lecture des métadonnées géométriques d'origine ---
        orig_img = nib.load(tmp_path)
        original_shape = orig_img.shape  # Récupère la taille réelle (ex: 512, 512, 75)
        original_affine = orig_img.affine # Récupère l'orientation spatiale du scanner

        # --- 5. Inférence par le modèle ---
        result = predict(tmp_path)
        mask = result["mask"]
        tumor_detected = result["tumor_detected"]

        # Plus besoin de zoom ni d'apply_orientation — Invertd s'en charge !
        liver_voxels = int(np.sum(mask == 1) + np.sum(mask == 2))
        tumor_voxels = int(np.sum(mask == 2))
        has_tumor = "True" if tumor_voxels > 0 or tumor_detected else "False"

        nib.save(
            nib.Nifti1Image(mask.astype(np.uint8), affine=original_affine),
            out_path
        )

        # --- 9. Envoi du fichier et des métadonnées ---
        return FileResponse(
            path=out_path,
            media_type="application/octet-stream",
            filename="segmentation_mask.nii",
            headers={
                "X-Has-Tumor": has_tumor,
                "X-Liver-Voxels": str(liver_voxels),
                "X-Tumor-Voxels": str(tumor_voxels)
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne lors de l'inférence ou du redimensionnement : {str(e)}"
        )
    
    finally:
        # Suppression du fichier d'entrée temporaire
        # Le fichier de sortie (out_path) est géré par FileResponse automatiquement
        if os.path.exists(tmp_path):
            os.remove(tmp_path)