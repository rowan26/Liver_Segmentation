from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
    ToTensord,
)
# Pas d'augmentation en inférence — uniquement les transforms déterministes
inference_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(
        keys=["image"],
        pixdim=(1.5, 1.5, 1.0),       # même résolution physique qu'en training
        mode="bilinear",
    ),
    Orientationd(
        keys=["image"],
        axcodes="RAS",                 # alignement standard
    ),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-200, a_max=200,         # fenêtre HU centrée sur le foie
        b_min=0.0,  b_max=1.0,
        clip=True,
    ),
    CropForegroundd(
        keys=["image"],
        source_key="image",            # crop basé sur l'image, pas le label
    ),
    Resized(
        keys=["image"],
        spatial_size=(128, 128, 96),   # shape fixe attendue par le modèle
    ),
    ToTensord(keys=["image"]),
])