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
    Invertd
)
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
    Invertd,
)

# Pas d'augmentation en inférence — uniquement les transforms déterministes
inference_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(
        keys=["image"],
        pixdim=(1.5, 1.5, 1.0),
        mode="bilinear",
    ),
    Orientationd(
        keys=["image"],
        axcodes="RAS",
    ),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-200, a_max=200,
        b_min=0.0,  b_max=1.0,
        clip=True,
    ),
    CropForegroundd(
        keys=["image"],
        source_key="image",
    ),
    Resized(
        keys=["image"],
        spatial_size=(128, 128, 96),
    ),
    ToTensord(keys=["image"]),
])

# Transform d'inversion — à appliquer sur le MASQUE PRÉDIT après inférence
# Elle rejoue inference_transforms en sens inverse
post_transforms = Compose([
    Invertd(
        keys="pred",                    # clé du masque prédit dans le dict
        transform=inference_transforms, # le pipeline dont on inverse les étapes
        orig_keys="image",               # clé de l'image originale (pour retrouver les métadonnées)
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=True,             # nearest neighbor pour préserver les classes {0,1,2}
        to_tensor=True,
    ),
])