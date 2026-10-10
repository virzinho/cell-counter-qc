"""Interactive research demo. Run: python -m streamlit run app.py"""
from dataclasses import asdict
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
import streamlit as st
from cell_counter import Config, segment, measure, overlay, normalize, quality
from cell_counter.dataset import read_image

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Nuclei Lab | Cell Counter QC", page_icon="🔬", layout="wide")
st.markdown("""<style>
.stApp {background:#f5f7fb;} h1,h2,h3 {color:#162334;}
[data-testid='stMetric'] {background:white;border:1px solid #dce4ec;border-radius:12px;padding:18px;}
[data-testid='stSidebar'] {background:#edf2f6;}
</style>""", unsafe_allow_html=True)
st.caption("BIOIMAGE ANALYSIS · REPRODUCIBLE RESEARCH DEMO")
st.title("Nuclei Lab")
st.write("Count fluorescent nuclei. Inspect their boundaries. Test what happens when image quality drops.")

with st.sidebar:
    st.header("Image & segmentation")
    source = st.radio("Image source", ["BBBC039 example", "Upload image"])
    method = st.selectbox("Method", ["watershed", "connected"], key="method")
    path = ROOT / "reports/selected_config.json"
    selected = json.loads(path.read_text())[method] if path.exists() else asdict(Config())
    min_area = st.slider("Minimum nucleus area (px²)", 10, 400, selected["min_area"], 10)
    distance = st.slider("Minimum peak distance (px)", 2, 25, selected["min_distance"])
    factor = st.slider("Otsu threshold multiplier", .3, 1.5, float(selected["threshold_factor"]), .05)
    st.caption("Defaults were selected on the official validation split. Changing sliders does not update the published benchmark.")
    st.divider()
    st.header("Image quality experiment")
    blur = st.slider("Gaussian blur (σ, pixels)", 0.0, 6.0, 0.0, .5)
    noise = st.slider("Gaussian noise (normalized intensity σ)", 0.0, .2, 0.0, .01)
    st.caption("Synthetic perturbations. They do not reproduce every microscope artifact.")

if source == "BBBC039 example":
    manifest = json.loads((ROOT / "examples/manifest.json").read_text())
    index = st.selectbox("Example", range(len(manifest)), format_func=lambda i: f"Validation image {i+1}", key="example")
    item = manifest[index]
    image = read_image(ROOT / "examples" / item["file"])
    st.caption(f"Source: BBBC039v1 · official validation split · reference annotation: {item['reference_count']} nuclei")
else:
    upload = st.file_uploader("Upload a single 2D fluorescent image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if upload is None:
        st.info("Upload an image or choose a bundled BBBC039 example.")
        st.stop()
    try:
        if upload.size > 10 * 1024**2:
            raise ValueError("Use an image smaller than 10 MB")
        with Image.open(upload) as opened:
            if opened.width * opened.height > 4_000_000:
                raise ValueError("Use an image with at most four million pixels")
            if getattr(opened, "n_frames", 1) != 1:
                raise ValueError("Export one 2D frame from the image stack first")
            image = np.array(opened)
        normalize(image)
    except Exception as error:
        st.error(f"Cannot process image: {error}")
        st.stop()

config = Config(min_area=min_area, min_distance=distance, threshold_factor=factor)
x = normalize(image)
perturbed = np.clip(gaussian_filter(x, blur) + np.random.default_rng(42).normal(0, noise, x.shape), 0, 1)
reference_prediction = segment(x, method, config)
prediction = segment(perturbed, method, config)
table = measure(prediction)
qc = quality(perturbed)
columns = st.columns(3)
columns[0].metric("Detected nuclei", len(table))
columns[1].metric("Count change from unperturbed", int(prediction.max() - reference_prediction.max()))
columns[2].metric("Median nucleus area", f"{table.area_px2.median():.0f} px²" if len(table) else "—")
left, right = st.columns(2)
left.image(perturbed, caption="Input after selected perturbations", clamp=True, width="stretch")
right.image(overlay(perturbed, prediction), caption="Detected nucleus boundaries", width="stretch")
tabs = st.tabs(["Measurements", "Quality indicators", "Published benchmark", "How it works"])
with tabs[0]:
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("Download measurements CSV", table.to_csv(index=False), "nuclei_measurements.csv", "text/csv")
    buf = io.BytesIO()
    Image.fromarray(prediction.astype(np.uint16)).save(buf, format="TIFF")
    st.download_button("Download instance labels TIFF", buf.getvalue(), "nuclei_labels.tif", "image/tiff")
    buf = io.BytesIO()
    Image.fromarray((overlay(perturbed, prediction)*255).astype(np.uint8)).save(buf, format="PNG")
    st.download_button("Download boundary overlay", buf.getvalue(), "nuclei_overlay.png", "image/png")
with tabs[1]:
    st.json(qc)
    st.write("These are descriptive indicators, not a validated quality score. Noise can increase Laplacian variance even when an image gets worse. Robust normalization removes pure intensity scaling, so dimmer images are not automatically marked as worse.")
with tabs[2]:
    chart = ROOT / "reports/robustness.png"
    if chart.exists():
        st.image(str(chart), width="stretch")
        st.dataframe(pd.read_csv(ROOT / "reports/summary.csv")[["method", "condition", "n_images", "absolute_count_error", "instance_f1", "foreground_dice"]], hide_index=True)
    st.caption("Benchmark uses frozen validation-selected parameters and the official test split. Interactive slider changes are exploratory.")
with tabs[3]:
    st.markdown("""1. Preserve 16-bit input and normalize intensities.
2. Smooth the image and threshold bright foreground.
3. Remove small regions and fill holes.
4. Either count connected regions or split them with distance-transform markers and watershed.
5. Measure each detected region in pixel units.

This is a **classical computer vision baseline**, with no trained neural network. It counts nuclei, not whole cells, and is intended for bright fluorescent nuclei on dark backgrounds. Touching, dim or irregular nuclei can be missed or split incorrectly. Physical units require microscope calibration.
""")
st.divider()
st.caption("Research and portfolio prototype · Not a clinical diagnostic tool · BBBC039 data: Broad Bioimage Benchmark Collection, CC0")
