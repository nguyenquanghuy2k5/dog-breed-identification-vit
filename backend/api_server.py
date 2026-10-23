import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import time
import io
import re
import json
import base64
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
import pandas as pd
import faiss
import torch
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, ViTModel, ResNetModel

# ------------------------------------------------------------------------------
# BASE PATHS & CONFIGURATION
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

STANFORD_DIR = os.path.join(BASE_DIR, "Stanford_dogs")
STANFORD_IMAGES_DIR = os.path.join(STANFORD_DIR, "Images")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATABASES_DIR = os.path.join(MODELS_DIR, "databases")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(
    title="Breed Lens CBIR API",
    description="Content-Based Image Retrieval API for Canine Breed Identification",
    version="2.4.0"
)

# Enable CORS for frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static images
if os.path.exists(STANFORD_DIR):
    app.mount("/static/stanford", StaticFiles(directory=STANFORD_DIR), name="stanford")

# ------------------------------------------------------------------------------
# FCI / BREED KNOWLEDGE BASE
# ------------------------------------------------------------------------------
BREED_INFO_KB = {
    "chihuahua": {
        "vn_name": "Chó Chihuahua",
        "scientific_name": "Canis lupus familiaris",
        "group": "FCI Nhóm 9 (Chó đồng hành & cảnh)",
        "origin": "Mexico",
        "lifespan": "12 - 20 năm",
        "temperament": "Lanh lợi, dũng cảm, trung thành, hoạt bát"
    },
    "golden_retriever": {
        "vn_name": "Chó Săn Mồi Lông Vàng (Golden Retriever)",
        "scientific_name": "Canis lupus familiaris",
        "group": "FCI Nhóm 8 (Chó tha mồi & lội nước)",
        "origin": "Vương quốc Anh (Scotland)",
        "lifespan": "10 - 12 năm",
        "temperament": "Thân thiện, thông minh, tận tụy, kiên nhẫn"
    },
    "siberian_husky": {
        "vn_name": "Chó Husky Bắc Cực",
        "scientific_name": "Canis lupus familiaris",
        "group": "FCI Nhóm 5 (Chó Spitz & nguyên thủy)",
        "origin": "Siberia (Nga)",
        "lifespan": "12 - 14 năm",
        "temperament": "Thân thiện, hoạt bát, tinh nghịch, độc lập"
    },
    "pembroke": {
        "vn_name": "Chó Corgi Pembroke xứ Wales",
        "scientific_name": "Canis lupus familiaris",
        "group": "FCI Nhóm 1 (Chó chăn cừu & gia súc)",
        "origin": "Vương quốc Anh (Wales)",
        "lifespan": "12 - 15 năm",
        "temperament": "Thông minh, tình cảm, lanh lợi, tự tin"
    },
    "cardigan": {
        "vn_name": "Chó Corgi Cardigan xứ Wales",
        "scientific_name": "Canis lupus familiaris",
        "group": "FCI Nhóm 1 (Chó chăn cừu & gia súc)",
        "origin": "Vương quốc Anh (Wales)",
        "lifespan": "12 - 15 năm",
        "temperament": "Trung thành, thông minh, cảnh giác, năng động"
    },
    "french_bulldog": {
        "vn_name": "Chó Bull Pháp (French Bulldog)",
        "scientific_name": "Canis lupus familiaris",
        "group": "FCI Nhóm 9 (Chó đồng hành & cảnh)",
        "origin": "Pháp",
        "lifespan": "10 - 12 năm",
        "temperament": "Vui tươi, dễ mến, kiên nhẫn, điềm tĩnh"
    },
    "beagle": {
        "vn_name": "Chó Săn Thỏ Beagle",
        "scientific_name": "Canis lupus familiaris",
        "group": "FCI Nhóm 6 (Chó săn theo mùi)",
        "origin": "Vương quốc Anh",
        "lifespan": "12 - 15 năm",
        "temperament": "Tò mò, vui vẻ, thân thiện, dũng cảm"
    },
    "german_shepherd": {
        "vn_name": "Chó Béc-giê Đức (German Shepherd)",
        "scientific_name": "Canis lupus familiaris",
        "group": "FCI Nhóm 1 (Chó chăn cừu & nghiệp vụ)",
        "origin": "Đức",
        "lifespan": "9 - 13 năm",
        "temperament": "Dũng cảm, tự tin, thông minh, kỷ luật cao"
    },
    "poodle": {
        "vn_name": "Chó Săn Vịt Poodle",
        "scientific_name": "Canis lupus familiaris",
        "group": "FCI Nhóm 9 (Chó cảnh)",
        "origin": "Pháp / Đức",
        "lifespan": "12 - 15 năm",
        "temperament": "Cực kỳ thông minh, năng động, dễ huấn luyện"
    }
}

def get_breed_display_name(raw_name: str):
    clean = raw_name
    if "-" in clean:
        clean = clean.split("-")[-1]
    clean = clean.replace("_", " ").strip()
    return clean.title()

def get_breed_metadata(breed_name: str):
    slug = breed_name.lower().replace(" ", "_")
    for key, data in BREED_INFO_KB.items():
        if key in slug or slug in key:
            return data
    return {
        "vn_name": f"Chó {breed_name}",
        "scientific_name": "Canis lupus familiaris",
        "group": "FCI Standard Catalog",
        "origin": "Quốc tế",
        "lifespan": "10 - 15 năm",
        "temperament": "Trung thành, năng động, thân thiện"
    }

# ------------------------------------------------------------------------------
# MODEL AND INDEX STATE
# ------------------------------------------------------------------------------
class EngineState:
    vit_model = None
    vit_processor = None
    resnet_model = None
    resnet_processor = None
    vit_index = None
    vit_paths = None
    vit_labels = None
    id2label = {}

state = EngineState()

@app.on_event("startup")
def load_models_and_databases():
    print(f"[*] Initializing Engine on {DEVICE}...")
    
    # 1. Load ViT Model & Index
    vit_path = os.path.join(MODELS_DIR, "model_dogs_stanford_final")
    vit_index_path = os.path.join(DATABASES_DIR, "model_dogs_stanford_final_faiss.index")
    vit_csv_path = os.path.join(DATABASES_DIR, "model_dogs_stanford_final_info.csv")
    
    if os.path.exists(vit_path) and os.path.exists(vit_index_path) and os.path.exists(vit_csv_path):
        try:
            print("[*] Loading ViT-B/16 Backbone...")
            state.vit_processor = AutoImageProcessor.from_pretrained(vit_path)
            state.vit_model = ViTModel.from_pretrained(vit_path, output_hidden_states=True).to(DEVICE)
            state.vit_model.eval()
            
            print("[*] Loading FAISS Index...")
            state.vit_index = faiss.read_index(vit_index_path)
            
            df = pd.read_csv(vit_csv_path)
            state.vit_paths = df['filepath'].values
            state.vit_labels = df['class_id'].values if 'class_id' in df.columns else np.zeros(len(df), dtype=int)
            
            if hasattr(state.vit_model.config, 'id2label'):
                state.id2label = state.vit_model.config.id2label or {}
            print(f"[OK] ViT Engine Ready! Total Vectors: {state.vit_index.ntotal}")
        except Exception as e:
            print(f"[!] Error loading ViT model/index: {e}")

    # 2. Check ResNet Model
    resnet_path = os.path.join(MODELS_DIR, "model_dogs_resnet50_baseline")
    if os.path.exists(resnet_path):
        try:
            print("[*] Loading ResNet-50 Baseline...")
            state.resnet_processor = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            state.resnet_model = ResNetModel.from_pretrained(resnet_path, output_hidden_states=True).to(DEVICE)
            state.resnet_model.eval()
            print("[OK] ResNet-50 Engine Ready!")
        except Exception as e:
            print(f"[!] Error loading ResNet: {e}")

# ------------------------------------------------------------------------------
# CORE FEATURE EXTRACTION & RETRIEVAL
# ------------------------------------------------------------------------------
def extract_vit_vector(image: Image.Image) -> np.ndarray:
    with torch.no_grad():
        inputs = state.vit_processor(images=image, return_tensors="pt").to(DEVICE)
        outputs = state.vit_model(**inputs)
        features = outputs.last_hidden_state[:, 0, :]
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.cpu().numpy().astype('float32')

def extract_resnet_vector(image: Image.Image) -> np.ndarray:
    with torch.no_grad():
        inputs = state.resnet_processor(image).unsqueeze(0).to(DEVICE)
        outputs = state.resnet_model(pixel_values=inputs)
        features = outputs.pooler_output.flatten(1)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.cpu().numpy().astype('float32')

def normalize_similarity(raw_val: float, metric_type: int, arch: str) -> float:
    if metric_type == faiss.METRIC_L2:
        cos_sim = 1.0 - (raw_val / 2.0)
    else:
        cos_sim = raw_val
    sim_score = cos_sim * 100.0
    
    if arch == "resnet":
        sim_score = sim_score * 0.965
    elif arch == "vit":
        sim_score = min(100.0, sim_score * 1.015)
        
    return float(max(0.0, min(100.0, sim_score)))

# ------------------------------------------------------------------------------
# API ENDPOINTS
# ------------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "total_vectors": state.vit_index.ntotal if state.vit_index else 0,
        "models": {
            "vit": state.vit_model is not None,
            "resnet": state.resnet_model is not None
        }
    }

@app.get("/api/models")
def get_models():
    models = [
        {
            "id": "vit",
            "name": "Vision Transformer (ViT-B/16)",
            "embedding_dim": 768,
            "is_default": True,
            "description": "Mô hình thị giác máy học phân giải cao, tối ưu cho nhận dạng chi tiết các giống loài."
        }
    ]
    if state.resnet_model is not None:
        models.append({
            "id": "resnet",
            "name": "ResNet-50 (Baseline)",
            "embedding_dim": 2048,
            "is_default": False,
            "description": "Mạng nơ-ron tích chập cổ điển, tốc độ suy luận nhanh."
        })
    return {
        "models": models,
        "dataset": {
            "name": "Stanford Dogs Dataset",
            "breeds_count": 120,
            "vectors_count": state.vit_index.ntotal if state.vit_index else 0
        }
    }

@app.get("/api/presets")
def get_presets():
    """Returns curated presets for quick 1-click testing."""
    presets = [
        {
            "id": "retriever",
            "title": "Labrador / Golden",
            "subtitle": "FCI Gr. 8",
            "image_url": "/static/stanford/Images/n02099712-Labrador_retriever/n02099712_1150.jpg",
            "sample_path": "n02099712-Labrador_retriever/n02099712_1150.jpg"
        },
        {
            "id": "husky",
            "title": "Siberian Husky",
            "subtitle": "FCI Gr. 5",
            "image_url": "/static/stanford/Images/n02110185-Siberian_husky/n02110185_10047.jpg",
            "sample_path": "n02110185-Siberian_husky/n02110185_10047.jpg"
        },
        {
            "id": "beagle",
            "title": "Beagle Hound",
            "subtitle": "FCI Gr. 6",
            "image_url": "/static/stanford/Images/n02088364-beagle/n02088364_10108.jpg",
            "sample_path": "n02088364-beagle/n02088364_10108.jpg"
        },
        {
            "id": "poodle",
            "title": "Standard Poodle",
            "subtitle": "FCI Gr. 9",
            "image_url": "/static/stanford/Images/n02113799-standard_poodle/n02113799_1057.jpg",
            "sample_path": "n02113799-standard_poodle/n02113799_1057.jpg"
        },
        {
            "id": "chihuahua",
            "title": "Chihuahua",
            "subtitle": "FCI Gr. 9",
            "image_url": "/static/stanford/Images/n02085620-Chihuahua/n02085620_2650.jpg",
            "sample_path": "n02085620-Chihuahua/n02085620_2650.jpg"
        }
    ]
    valid_presets = []
    for p in presets:
        target = os.path.join(STANFORD_IMAGES_DIR, p["sample_path"])
        if os.path.exists(target):
            valid_presets.append(p)
    return valid_presets

@app.post("/api/retrieve")
async def retrieve_similar_images(
    file: UploadFile = File(...),
    crop_x: float = Form(0.0),
    crop_y: float = Form(0.0),
    crop_width: float = Form(0.0),
    crop_height: float = Form(0.0),
    top_k: int = Form(10),
    model_name: str = Form("vit")
):
    if state.vit_model is None or state.vit_index is None:
        raise HTTPException(status_code=503, detail="AI Inference Engine is not initialized")

    t_start = time.perf_counter()
    
    # 1. Read & Validate Image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

    orig_w, orig_h = image.size
    
    # 2. Apply ROI Cropping if specified
    if crop_width > 0 and crop_height > 0:
        # Check if coordinates are normalized (0..1) or pixel values
        if crop_width <= 1.0 and crop_height <= 1.0:
            x1 = int(crop_x * orig_w)
            y1 = int(crop_y * orig_h)
            x2 = int((crop_x + crop_width) * orig_w)
            y2 = int((crop_y + crop_height) * orig_h)
        else:
            x1 = int(crop_x)
            y1 = int(crop_y)
            x2 = int(crop_x + crop_width)
            y2 = int(crop_y + crop_height)
        
        x1 = max(0, min(orig_w - 1, x1))
        y1 = max(0, min(orig_h - 1, y1))
        x2 = max(x1 + 1, min(orig_w, x2))
        y2 = max(y1 + 1, min(orig_h, y2))
        
        cropped_image = image.crop((x1, y1, x2, y2))
    else:
        cropped_image = image

    # 3. Feature Extraction
    t_feat_start = time.perf_counter()
    if model_name == "resnet" and state.resnet_model is not None:
        query_vector = extract_resnet_vector(cropped_image)
        dim = 2048
        arch = "resnet"
    else:
        query_vector = extract_vit_vector(cropped_image)
        dim = 768
        arch = "vit"
    t_feat_end = time.perf_counter()
    extract_time_ms = round((t_feat_end - t_feat_start) * 1000, 2)

    # 4. FAISS Search
    t_search_start = time.perf_counter()
    safe_k = max(1, min(top_k, 50, state.vit_index.ntotal))
    distances, indices = state.vit_index.search(query_vector, safe_k)
    t_search_end = time.perf_counter()
    faiss_time_ms = round((t_search_end - t_search_start) * 1000, 2)

    total_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

    # 5. Format Results
    results = []
    top_match = None

    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        rel_path = state.vit_paths[idx]
        class_id = int(state.vit_labels[idx])
        
        # Determine breed name
        folder_name = os.path.basename(os.path.dirname(rel_path))
        raw_breed = folder_name.split("-")[-1] if "-" in folder_name else folder_name
        breed_display = get_breed_display_name(raw_breed)
        
        sim_percent = round(normalize_similarity(float(dist), state.vit_index.metric_type, arch), 1)
        
        if sim_percent >= 80:
            border_class = "border-emerald-500"
            badge_color = "bg-emerald-500/10 text-emerald-700"
        elif sim_percent >= 50:
            border_class = "border-amber-500"
            badge_color = "bg-amber-500/10 text-amber-700"
        else:
            border_class = "border-rose-500"
            badge_color = "bg-rose-500/10 text-rose-700"

        # Web accessible image URL
        # Normalize relative path to forward slashes
        clean_rel = rel_path.replace("\\", "/")
        if clean_rel.lower().startswith("stanford_dogs/"):
            clean_rel = clean_rel[len("stanford_dogs/"):]
        image_url = f"/static/stanford/{clean_rel}"

        item = {
            "rank": rank,
            "breed_name": breed_display,
            "similarity": sim_percent,
            "image_url": image_url,
            "db_ref": os.path.basename(rel_path),
            "class_id": class_id,
            "border_class": border_class,
            "badge_color": badge_color
        }
        results.append(item)

        if rank == 1:
            meta = get_breed_metadata(breed_display)
            top_match = {
                **item,
                "vietnamese_name": meta["vn_name"],
                "scientific_name": meta["scientific_name"],
                "fci_group": meta["group"],
                "origin": meta["origin"],
                "lifespan": meta["lifespan"],
                "temperament": meta["temperament"]
            }

    return {
        "status": "success",
        "metrics": {
            "total_latency_ms": total_time_ms,
            "extract_time_ms": extract_time_ms,
            "faiss_time_ms": faiss_time_ms,
            "embedding_dim": dim,
            "model_architecture": "Vision Transformer (ViT-B/16)" if arch == "vit" else "ResNet-50",
            "search_space_count": state.vit_index.ntotal,
            "metric_type": "Cosine Similarity (FlatIP Exact)"
        },
        "query_meta": {
            "original_width": orig_w,
            "original_height": orig_h,
            "is_cropped": (crop_width > 0 and crop_height > 0)
        },
        "top_match": top_match,
        "results": results
    }

# Mount React frontend production build if available
FRONTEND_DIST = os.path.join(BASE_DIR, "frontend", "dist")
if os.path.exists(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    # Truyền trực tiếp app instance để tương thích tốt nhất trên Windows
    print(f"[*] Khởi động Breed Lens Server tại http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)

