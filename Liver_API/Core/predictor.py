# core/predictor.py
# Responsabilité unique : charger le modèle U-Net en mémoire

import torch
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from Core.transforms import inference_transforms
import numpy as np

# --- Chemin vers le checkpoint Run 4 ---
MODEL_PATH = "model/best_metric_model.pth"

# --- Device : GPU si disponible, sinon CPU ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> torch.nn.Module:
    """
    Instancie le U-Net avec exactement les mêmes paramètres
    qu'en training, puis charge les poids sauvegardés.
    """
    model = UNet(
        spatial_dims=3,
        in_channels=1,        # scanner CT = 1 canal (niveaux de gris)
        out_channels=3,       # 0=background, 1=foie, 2=tumeur
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(DEVICE)

    # Charge les poids — map_location gère CPU/GPU automatiquement
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    # Mode inférence : fige BatchNorm, désactive Dropout
    model.eval()

    return model

def predict(nifti_path: str) -> dict:
    """
    Reçoit le chemin vers un fichier .nii.gz,
    retourne le masque prédit et la détection de tumeur.
    """

    # --- 1. Preprocessing ---
    data  = inference_transforms({"image": nifti_path})
    image = data["image"].unsqueeze(0).to(DEVICE)
    print(f"[DEBUG] image après transforms : {image.shape}")

    # --- 2. Inférence ---
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=image,
            roi_size=(128, 128, 96),
            sw_batch_size=1,
            predictor=model,
        )
    print(f"[DEBUG] logits shape : {logits.shape}")

    # --- 3. Conversion ---
    probs = torch.softmax(logits, dim=1)
    print(f"[DEBUG] probs shape  : {probs.shape}")

    mask = torch.argmax(probs, dim=1)
    print(f"[DEBUG] mask shape avant squeeze : {mask.shape}")

    if isinstance(mask, (list, tuple)):
        mask = mask[0]
    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze(0).cpu().numpy()

    print(f"[DEBUG] mask shape finale : {mask.shape}")

    tumor_detected = bool(np.any(mask == 2))
    return {"mask": mask, "tumor_detected": tumor_detected}

    return{
        "mask": mask.tolist(),  # conversion en liste pour JSON
        "tumor_detected": tumor_detected
    }


# Chargement unique au démarrage — cette variable est importée par app.py
model = load_model()
print(f"✅ Modèle chargé sur {DEVICE}")