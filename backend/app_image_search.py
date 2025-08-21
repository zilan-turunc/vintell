import os
import streamlit as st
import faiss
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Paths
IMG_FOLDER = "rag/images"
INDEX_PATH = "rag/img_index/index.faiss"
META_PATH = "rag/img_index/meta.json"
EMBED_MODEL = "clip-ViT-B-32"

# Load model and index
st.set_page_config(page_title="Vintell Image Search", layout="wide")
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    raw_meta = json.load(f)

# If each item is a dict with "path", extract just the path string
img_paths = [entry["path"] if isinstance(entry, dict) else entry for entry in raw_meta]

# Embed query image using CLIP
def embed_image(image: Image.Image):
    from transformers import CLIPProcessor, CLIPModel
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
    return image_features.numpy()

# Sidebar UI
st.sidebar.title("Upload a Fashion Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption="Query Image", use_column_width=True)
    query_emb = embed_image(image)
    D, I = index.search(query_emb, 5)

    st.header("🔍 Top Matching Results")
    cols = st.columns(5)

    for rank, (idx, dist) in enumerate(zip(I[0], D[0])):
        if idx < len(img_paths):
            match_path = img_paths[idx]
            try:
                result_image = Image.open(match_path)
                with cols[rank % 5]:
                    st.image(result_image, caption=f"Score: {1 - dist:.2f}", use_column_width=True)
            except Exception as e:
                st.warning(f"❌ Could not load image: {match_path} ({e})")
else:
    st.info("📎 Upload an image using the sidebar to begin.")
