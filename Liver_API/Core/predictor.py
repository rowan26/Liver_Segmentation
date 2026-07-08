import torch
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from Core.transforms import inference_transforms
import numpy as np
import gc


MODEL_PATH = "model/best_metric_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model() -> torch.nn.Module:

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(DEVICE)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model



def predict(nifti_path: str) -> dict:

    print("\n" + "=" * 60)
    print("[PREDICTOR] Fichier :", nifti_path)


    data = inference_transforms(
        {
            "image": nifti_path
        }
    )


    image = (
        data["image"]
        .unsqueeze(0)
        .to(DEVICE)
    )


    print(
        f"[PREDICTOR] Tensor : {image.shape}"
    )


    with torch.no_grad():

        logits = model(image)


    probs = torch.softmax(
        logits,
        dim=1
    )


    mask = (
        torch.argmax(
            probs,
            dim=1
        )
        .squeeze(0)
        .cpu()
        .numpy()
    )


    print(
        "[PREDICTOR] Classes:",
        np.unique(mask)
    )


    tumor_detected = bool(
        np.any(mask == 2)
    )


    # Libération mémoire
    del image
    del logits
    del probs
    del data

    gc.collect()


    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


    print("=" * 60)


    return {
        "mask": mask,
        "tumor_detected": tumor_detected
    }



model = load_model()

print(
    f"✅ Modèle chargé sur {DEVICE}"
)