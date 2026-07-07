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
    # Preprocessing
    data  = inference_transforms({"image": nifti_path})
    image = data["image"].unsqueeze(0).to(DEVICE)

    # Inférence directe — le modèle attend un volume entier resizé
    with torch.no_grad():
        logits = model(image)

    # Conversion logits → masque
    mask = torch.argmax(
        torch.softmax(logits, dim=1),
        dim=1
    ).squeeze(0).cpu().numpy()

    return {
        "mask": mask,
        "tumor_detected": bool(np.any(mask == 2))
    }

# Chargement unique au démarrage — cette variable est importée par app.py
model = load_model()
print(f"✅ Modèle chargé sur {DEVICE}")