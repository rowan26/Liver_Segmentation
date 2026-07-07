import streamlit as st
import requests
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import tempfile

st.set_page_config(page_title="AI Liver Seg", layout="wide")
st.title("🩻 Segmentation Hépatique & Tumorale 3D")

API_URL = "http://localhost:8000/predict"

uploaded_file = st.file_uploader("Chargez un scanner CT (.nii ou .nii.gz)", type=["nii", "nii.gz"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info("Analyse du volume en cours...")
        
        with st.spinner("Inférence par le modèle UNet 3D en cours..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
            
            try:
                # 1. Sauvegarde locale temporaire du fichier d'entrée pour Nibabel
                with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp_in:
                    tmp_in.write(uploaded_file.getvalue())
                    tmp_in_path = tmp_in.name
                
                original_img = nib.load(tmp_in_path)
                original_data = original_img.get_fdata()
                os.remove(tmp_in_path) # Nettoyage immédiat
                
                # 2. Requête vers l'API
                response = requests.post(API_URL, files=files)
                response.raise_for_status()
                
                # 3. Récupération des headers synchronisés
                has_tumor = response.headers.get("X-Has-Tumor", "False") == "True"
                liver_voxels = int(response.headers.get("X-Liver-Voxels", 0))
                tumor_voxels = int(response.headers.get("X-Tumor-Voxels", 0))
                
                # 4. Sauvegarde locale temporaire du masque reçu
                with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp_out:
                    tmp_out.write(response.content)
                    tmp_out_path = tmp_out.name
                    
                mask_img = nib.load(tmp_out_path)
                mask_data = mask_img.get_fdata()
                os.remove(tmp_out_path) # Nettoyage immédiat
                
                # 5. Interface utilisateur (Bannières & Métriques)
                if has_tumor:
                    st.error("🔴 ALERTE : Tumeur détectée")
                else:
                    st.success("🟢 SAIN : Aucune tumeur détectée")
                    
                st.markdown("### 📊 Statistiques Volumétriques")
                total_voxels = original_data.size
                
                metric_col1, metric_col2 = st.columns(2)
                metric_col1.metric("Volume Foie (Voxels)", f"{liver_voxels:,}")
                metric_col2.metric("Volume Tumeur (Voxels)", f"{tumor_voxels:,}")
                
                if total_voxels > 0:
                    st.progress(min(liver_voxels / total_voxels, 1.0), text=f"Foie : {(liver_voxels/total_voxels)*100:.2f}% de l'image")
                if has_tumor and liver_voxels > 0:
                    st.progress(min(tumor_voxels / liver_voxels, 1.0), text=f"Tumeur : {(tumor_voxels/liver_voxels)*100:.2f}% du Foie")

            except Exception as e:
                st.error(f"Erreur lors du traitement : {e}")
                st.stop()

  # 6. Visualiseur Coupe par Coupe (Z-Axis)
    with col2:
        st.markdown("### 👁️ Visualisation Coupe par Coupe")
        
        max_slices = original_data.shape[2]
        slice_idx = st.slider("Sélectionnez la coupe axiale (Z)", 0, max_slices - 1, max_slices // 2)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis('off')
        
        # 1. Extraction et affichage du scanner original
        base_slice = original_data[:, :, slice_idx].T
        vmin, vmax = np.percentile(base_slice, [2, 98])
        ax.imshow(base_slice, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        
        # --- RÉALIGNEMENT GÉOMÉTRIQUE STRICT ET DÉFINITIF ---
        # 1. On oriente les axes de la matrice comme l'image du scanner
        # ✅ Le masque est déjà dans le bon espace
        mask_slice = mask_data[:, :, slice_idx].T


        # ----------------------------------------------------

        # 3. Création de la couche de couleur transparente (RGBA)
        overlay = np.zeros(mask_slice.shape + (4,))
        
        # Classe 1 (Foie) en Bleu, Classe 2 (Tumeur) en Rouge
        overlay[mask_slice == 1] = [0, 0.5, 1, 0.4]  # 40% de transparence
        overlay[mask_slice == 2] = [1, 0, 0, 0.7]    # 70% de transparence
        # 4. Superposition
        ax.imshow(overlay, origin='lower')
        st.pyplot(fig)