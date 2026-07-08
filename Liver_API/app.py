
import shutil
import uuid
import tempfile
import os
import gc

import nibabel as nib
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from Core.predictor import DEVICE, predict

from scipy.ndimage import zoom

from nibabel.orientations import (
    io_orientation,
    axcodes2ornt,
    ornt_transform,
    apply_orientation
)


app = FastAPI(
    title="Liver Segmentation API",
    description="3D U-Net — liver & tumor segmentation from NIfTI CT scans",
    version="1.0.0"
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE)
    }



def delete_file(path):
    """
    Suppression sécurisée après envoi HTTP
    """
    try:
        if os.path.exists(path):
            os.remove(path)
            print("[APP] Supprimé :", path)
    except Exception as e:
        print("[APP] Suppression impossible :", e)



@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...)
):

    if not file.filename.endswith(
        (".nii", ".nii.gz")
    ):
        raise HTTPException(
            status_code=400,
            detail="Format invalide"
        )


    tmp_dir = tempfile.gettempdir()


    tmp_path = os.path.join(
        tmp_dir,
        f"{uuid.uuid4()}_{file.filename}"
    )


    out_path = os.path.join(
        tmp_dir,
        f"{uuid.uuid4()}_mask.nii"
    )


    # Sauvegarde upload
    with open(
        tmp_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )



    try:

        print("\n" + "#"*60)
        print("[APP] Chargement :", tmp_path)



        # ===============================
        # Lecture CT sans garder handle
        # ===============================

        img = nib.load(
            tmp_path
        )


        original_shape = img.shape

        original_affine = img.affine.copy()

        original_data = (
            img
            .get_fdata()
            .copy()
        )


        del img
        gc.collect()



        print(
            "[APP] Shape :",
            original_shape
        )


        print(
            "[APP] Min Max :",
            original_data.min(),
            original_data.max()
        )



        # ===============================
        # Orientation originale
        # ===============================

        orig_ornt = io_orientation(
            original_affine
        )


        print(
            "[APP] Orientation :",
            nib.orientations.ornt2axcodes(
                orig_ornt
            )
        )



        # ===============================
        # Prediction
        # ===============================

        result = predict(
            tmp_path
        )


        mask = result["mask"]

        tumor_detected = result[
            "tumor_detected"
        ]


        mask = np.asarray(
            mask
        )


        print(
            "[APP] Mask modèle :",
            mask.shape,
            np.unique(mask)
        )



        # ===============================
        # RAS -> orientation originale
        # ===============================

        ras_ornt = axcodes2ornt(
            (
                "R",
                "A",
                "S"
            )
        )


        transform = ornt_transform(
            ras_ornt,
            orig_ornt
        )


        mask = apply_orientation(
            mask,
            transform
        )


        print(
            "[APP] Après orientation :",
            mask.shape
        )



        # ===============================
        # Resize
        # ===============================

        zoom_factor = [

            original_shape[0]
            /
            mask.shape[0],

            original_shape[1]
            /
            mask.shape[1],

            original_shape[2]
            /
            mask.shape[2]
        ]


        mask = zoom(
            mask,
            zoom_factor,
            order=0
        )



        print(
            "[APP] Après resize :",
            mask.shape
        )



        # ===============================
        # Statistiques
        # ===============================

        liver_voxels = int(
            np.sum(mask == 1)
            +
            np.sum(mask == 2)
        )


        tumor_voxels = int(
            np.sum(mask == 2)
        )


        has_tumor = (
            "True"
            if tumor_voxels > 0
            or tumor_detected
            else
            "False"
        )



        # ===============================
        # Sauvegarde NIFTI
        # ===============================

        output_img = nib.Nifti1Image(
            mask.astype(
                np.uint8
            ),
            original_affine
        )


        nib.save(
            output_img,
            out_path
        )


        del output_img
        gc.collect()



        print(
            "[APP] Foie:",
            liver_voxels
        )

        print(
            "[APP] Tumeur:",
            tumor_voxels
        )

        print("#"*60)



        return FileResponse(

            path=out_path,

            media_type=
            "application/octet-stream",

            filename=
            "segmentation_mask.nii",


            background=
            BackgroundTask(
                delete_file,
                out_path
            ),


            headers={

                "X-Has-Tumor":
                has_tumor,


                "X-Liver-Voxels":
                str(
                    liver_voxels
                ),


                "X-Tumor-Voxels":
                str(
                    tumor_voxels
                )
            }
        )



    except Exception as e:

        import traceback

        traceback.print_exc()


        raise HTTPException(

            status_code=500,

            detail=str(e)
        )



    finally:

        gc.collect()


        delete_file(
            tmp_path
        )