import streamlit as st
import numpy as np
import cv2
import math
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from PIL import Image

def preprocess_image_pil(pil_img, max_side=800):
    """Redimensiona mantendo proporção para acelerar processamento"""
    w, h = pil_img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        pil_img = pil_img.resize(new_size, resample=Image.LANCZOS)
    return pil_img


def extract_pixels(pil_img, sample_size=30000, remove_transparent=True, remove_whites=True):
    """Retorna array Nx3 em BGR (OpenCV) pronto para clustering"""
    img = np.array(pil_img)
    if img.shape[-1] == 4 and remove_transparent:
        alpha = img[..., 3]
        mask = alpha > 10
        img = img[mask]
    else:
        img = img.reshape(-1, img.shape[-1])
    if img.shape[-1] == 4:
        img = img[:, :3]
    img_bgr = img[..., ::-1].astype(np.uint8)

    # Remover quase-brancos/quase-pretos
    if remove_whites:
        lab = cv2.cvtColor(img_bgr.reshape(-1,1,3), cv2.COLOR_BGR2LAB).reshape(-1,3)
        L = lab[:, 0]
        mask_keep = (L > 5) & (L < 250)
        img_bgr = img_bgr[mask_keep]

    # Amostragem aleatória para velocidade
    n_pixels = img_bgr.shape[0]
    if n_pixels > sample_size:
        idx = np.random.choice(n_pixels, sample_size, replace=False)
        img_sample = img_bgr[idx]
    else:
        img_sample = img_bgr
    return img_sample


def cluster_colors(img_pixels_bgr, n_colors=6, use_mini_batch=True, n_init=10, use_gmm=False):
    """Clusterização de cores no espaço LAB"""
    lab = cv2.cvtColor(img_pixels_bgr.reshape(-1,1,3), cv2.COLOR_BGR2LAB).reshape(-1,3).astype(np.float32)

    if use_gmm:
        gm = GaussianMixture(n_components=n_colors, covariance_type='tied', random_state=42)
        gm.fit(lab)
        labels = gm.predict(lab)
        centers_lab = gm.means_
    else:
        if use_mini_batch:
            km = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init=n_init, batch_size=4096)
        else:
            km = KMeans(n_clusters=n_colors, random_state=42, n_init=n_init)
        labels = km.fit_predict(lab)
        centers_lab = km.cluster_centers_

    centers_lab_uint8 = centers_lab.astype(np.uint8).reshape(-1,1,3)
    centers_bgr = cv2.cvtColor(centers_lab_uint8, cv2.COLOR_LAB2BGR).reshape(-1,3)
    centers_rgb = centers_bgr[..., ::-1]
    unique, counts = np.unique(labels, return_counts=True)
    counts_full = np.zeros(n_colors, dtype=int)
    counts_full[unique] = counts
    order = np.argsort(-counts_full)
    centers_rgb = centers_rgb[order]
    counts_full = counts_full[order]
    centers_rgb = np.clip(centers_rgb, 0, 255).astype(int)
    return centers_rgb, counts_full


def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

st.set_page_config(page_title="🎨 ColorMind — Gerador de Paletas", page_icon="🎨", layout="wide")

st.title("🎨 ColorMind — Gerador de Paletas Inteligente")
st.write("Envie uma imagem e descubra as cores dominantes com precisão perceptual.")

uploaded_file = st.file_uploader("📸 Faça upload de uma imagem", type=["jpg", "jpeg", "png"])

st.sidebar.header("⚙️ Configurações")
num_colors = st.sidebar.slider("Número de cores", 3, 24, 6)
precision = st.sidebar.slider("Precisão (tamanho da amostragem)", 10000, 80000, 40000, step=5000)
max_side = st.sidebar.slider("Tamanho máximo do lado da imagem", 400, 1600, 900, step=100)
algorithm = st.sidebar.selectbox("Algoritmo de Clusterização", ["KMeans", "MiniBatchKMeans", "GaussianMixture"])
remove_whites = st.sidebar.checkbox("Remover brancos/preto (fundo)", True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem carregada", use_container_width=True)

    with st.spinner("🔍 Analisando cores..."):
        pil_small = preprocess_image_pil(image, max_side=max_side)
        pixels = extract_pixels(pil_small, sample_size=precision, remove_transparent=True, remove_whites=remove_whites)

        use_gmm = (algorithm == "GaussianMixture")
        use_mini_batch = (algorithm == "MiniBatchKMeans")

        centers_rgb, counts = cluster_colors(
            pixels,
            n_colors=num_colors,
            use_mini_batch=use_mini_batch,
            n_init=15,
            use_gmm=use_gmm,
        )

        hex_colors = [rgb_to_hex(c) for c in centers_rgb]
        total = counts.sum()
        percentages = [(c / total * 100) for c in counts]

    st.subheader("🎨 Paleta gerada")

    cols_per_row = 5
    rows = math.ceil(len(hex_colors) / cols_per_row)

    for i in range(rows):
        row_colors = hex_colors[i * cols_per_row:(i + 1) * cols_per_row]
        row_perc = percentages[i * cols_per_row:(i + 1) * cols_per_row]
        cols = st.columns(len(row_colors))
        for col, hex_code, perc in zip(cols, row_colors, row_perc):
            with col:
                st.markdown(
                    f"""
                    <div style="
                        background-color:{hex_code};
                        height:100px;
                        border-radius:10px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        color:#000000aa;
                        font-weight:bold;
                        margin-bottom:5px;
                        box-shadow:0 2px 4px rgba(0,0,0,0.2);
                        transition:transform 0.2s ease;
                    " 
                    onmouseover="this.style.transform='scale(1.05)';" 
                    onmouseout="this.style.transform='scale(1)';">
                    </div>
                    <p style="text-align:center; font-family:monospace;">{hex_code}<br><span style='font-size:11px; color:#555;'>({perc:.1f}%)</span></p>
                    """,
                    unsafe_allow_html=True
                )

    st.success("✅ Paleta gerada com sucesso!")

else:
    st.info("Envie uma imagem para começar.")
