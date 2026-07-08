
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    Resized,
    ToTensord,
)

import numpy as np


class DebugTransform:

    def __init__(self, name):
        self.name = name

    def __call__(self, data):

        img = data["image"]

        if hasattr(img, "shape"):

            print(
                f"[TRANSFORM] {self.name:30s}"
                f" shape={img.shape}"
                f" min={float(img.min()):.3f}"
                f" max={float(img.max()):.3f}"
            )

        return data



inference_transforms = Compose([


    # ============================
    # Chargement NIFTI
    # ============================

    LoadImaged(

        keys=["image"],

        image_only=False

    ),


    DebugTransform(
        "LoadImaged"
    ),



    # ============================
    # Ajout canal CT
    # ============================

    EnsureChannelFirstd(

        keys=["image"],

        channel_dim="no_channel"

    ),


    DebugTransform(
        "EnsureChannelFirstd"
    ),



    # ============================
    # Normalisation espace voxel
    # ============================

    Spacingd(

        keys=["image"],

        pixdim=(1.5,1.5,1.0),

        mode="bilinear"

    ),


    DebugTransform(
        "Spacingd"
    ),




    # ============================
    # Orientation standard réseau
    # ============================

    Orientationd(

        keys=["image"],

        axcodes="RAS"

    ),


    DebugTransform(
        "Orientationd"
    ),




    # ============================
    # Fenêtre CT foie
    # ============================

    ScaleIntensityRanged(

        keys=["image"],

        a_min=-200,

        a_max=200,

        b_min=0.0,

        b_max=1.0,

        clip=True

    ),


    DebugTransform(
        "ScaleIntensity"
    ),





    # ============================
    # Taille entrée UNet
    # ============================

    Resized(

        keys=["image"],

        spatial_size=(128,128,96)

    ),


    DebugTransform(
        "Resized"
    ),




    ToTensord(

        keys=["image"]

    )

])

