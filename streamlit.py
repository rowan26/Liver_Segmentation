
import streamlit as st
import requests
import nibabel as nib
import numpy as np
import tempfile
import os
import gc

from niivue_component import niivue_viewer


st.set_page_config(
    page_title="AI Liver Seg",
    layout="wide"
)


st.title(
    "🩻 Segmentation Hépatique & Tumorale 3D"
)


API_URL = "http://localhost:8000/predict"



uploaded_file = st.file_uploader(
    "Chargez un scanner CT (.nii ou .nii.gz)",
    type=[
        "nii",
        "nii.gz"
    ]
)



if uploaded_file is not None:


    ct_bytes = uploaded_file.getvalue()


    mask_bytes = None

    viewer_ready = False



    col1, col2 = st.columns(
        [
            1,
            2
        ]
    )



    with col1:


        st.info(
            "Analyse du volume en cours..."
        )


        with st.spinner(
            "Inférence UNet 3D..."
        ):


            tmp_in_path = None
            tmp_out_path = None



            try:


                # =====================
                # Sauvegarde CT local
                # =====================

                suffix = (
                    ".nii.gz"
                    if uploaded_file.name.endswith(".nii.gz")
                    else ".nii"
                )


                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as tmp:


                    tmp.write(
                        ct_bytes
                    )

                    tmp_in_path = tmp.name




                # =====================
                # Lecture CT
                # =====================

                img = nib.load(
                    tmp_in_path
                )


                original_data = (
                    img
                    .get_fdata()
                    .copy()
                )


                del img

                gc.collect()



                # =====================
                # API FastAPI
                # =====================

                files = {

                    "file":
                    (
                        uploaded_file.name,

                        ct_bytes,

                        "application/octet-stream"
                    )
                }



                response = requests.post(

                    API_URL,

                    files=files,

                    timeout=900
                )



                if response.status_code != 200:


                    st.error(
                        "Erreur FastAPI"
                    )


                    st.code(
                        response.text
                    )


                    st.stop()



                mask_bytes = response.content



                viewer_ready = True




                # =====================
                # Headers
                # =====================


                has_tumor = (

                    response.headers
                    .get(
                        "X-Has-Tumor",
                        "False"
                    )

                    ==
                    "True"
                )


                liver_voxels = int(

                    response.headers
                    .get(
                        "X-Liver-Voxels",
                        0
                    )

                )


                tumor_voxels = int(

                    response.headers
                    .get(
                        "X-Tumor-Voxels",
                        0
                    )

                )




                # =====================
                # Sauvegarde masque
                # =====================

                with tempfile.NamedTemporaryFile(

                    delete=False,

                    suffix=".nii"

                ) as tmp_out:


                    tmp_out.write(
                        mask_bytes
                    )


                    tmp_out_path = tmp_out.name




                mask_img = nib.load(
                    tmp_out_path
                )


                mask_data = (
                    mask_img
                    .get_fdata()
                    .copy()
                )


                del mask_img

                gc.collect()




                # =====================
                # Résultats
                # =====================


                if has_tumor:

                    st.error(
                        "🔴 ALERTE : Tumeur détectée"
                    )

                else:

                    st.success(
                        "🟢 SAIN : Aucune tumeur détectée"
                    )




                st.markdown(
                    "## 📊 Statistiques"
                )



                c1, c2 = st.columns(2)



                c1.metric(

                    "Foie (voxels)",

                    f"{liver_voxels:,}"
                )



                c2.metric(

                    "Tumeur (voxels)",

                    f"{tumor_voxels:,}"
                )




                total_voxels = (
                    original_data.size
                )



                if total_voxels > 0:


                    st.progress(

                        min(

                            liver_voxels /
                            total_voxels,

                            1.0

                        ),

                        text=(

                            f"Foie : "

                            f"{100*liver_voxels/total_voxels:.2f}%"

                        )

                    )



                if (

                    has_tumor

                    and

                    liver_voxels > 0

                ):


                    st.progress(

                        min(

                            tumor_voxels /
                            liver_voxels,

                            1.0

                        ),

                        text=(

                            f"Tumeur : "

                            f"{100*tumor_voxels/liver_voxels:.2f}% du foie"

                        )

                    )





            except Exception as e:


                import traceback


                st.error(
                    f"Erreur traitement : {e}"
                )


                st.code(
                    traceback.format_exc()
                )



            finally:


                gc.collect()


                for p in [

                    tmp_in_path,

                    tmp_out_path

                ]:


                    if p and os.path.exists(p):


                        try:

                            os.remove(p)

                        except:

                            pass






    # ==========================
    # NiiVue
    # ==========================

    with col2:


        st.markdown(
            "## 🧠 Visualisation 3D NiiVue"
        )



        if viewer_ready and mask_bytes is not None:


            viewer_state = niivue_viewer(

                nifti_data=ct_bytes,

                filename=uploaded_file.name,


                overlays=[

                    {

                        "data":
                        mask_bytes,


                        "filename":
                        "mask.nii"

                    }

                ],


                height=700,


                view_mode="multiplanar"

            )



            if viewer_state and "voxel" in viewer_state:


                st.info(

                    f"📍 Slice : {viewer_state['voxel'][2]}"

                )


        else:


            st.warning(
                "La visualisation sera disponible après une prédiction réussie."
            )
