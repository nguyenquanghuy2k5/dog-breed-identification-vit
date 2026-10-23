import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import os
import torch
import numpy as np
import pandas as pd
import faiss
import scipy.io
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, ResNetModel, ViTImageProcessor, ViTModel

ImageFile.LOAD_TRUNCATED_IMAGES = True
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# DATASET CLASSES (Modified to return filepath)
# ==============================================================================

class CUB200Dataset(Dataset):
    def __init__(self, root_dir, split="test", processor=None):
        self.processor = processor
        self.images_dir = os.path.join(root_dir, "images")
        
        # Đọc dữ liệu bằng Pandas theo yêu cầu
        images = pd.read_csv(os.path.join(root_dir, "images.txt"), sep=" ", names=["img_id", "filepath"])
        labels = pd.read_csv(os.path.join(root_dir, "image_class_labels.txt"), sep=" ", names=["img_id", "target"])
        split_df = pd.read_csv(os.path.join(root_dir, "train_test_split.txt"), sep=" ", names=["img_id", "is_training_img"])
        bboxes = pd.read_csv(os.path.join(root_dir, "bounding_boxes.txt"), sep=" ", names=["img_id", "x", "y", "width", "height"])
        
        data = images.merge(labels, on="img_id").merge(split_df, on="img_id").merge(bboxes, on="img_id")
        
        # Chỉ lấy tập Test (is_training_img == 0) làm Gallery (hoặc train tùy cấu hình)
        target_split = 1 if split == "train" else 0
        self.data = data[data["is_training_img"] == target_split].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.images_dir, row["filepath"])
        try:
            image = Image.open(img_path).convert("RGB")
            # Cắt Bounding Box
            x, y, w, h = row["x"], row["y"], row["width"], row["height"]
            image = image.crop((x, y, x + w, y + h))
        except Exception as e:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
            
        anchor_label = int(row["target"]) - 1 # 0-indexed
        
        if self.processor:
            pixel_values = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        else:
            from torchvision.transforms import Compose, Resize, ToTensor, Normalize
            tf = Compose([Resize((224, 224)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            pixel_values = tf(image)
            
        return {"pixel_values": pixel_values, "labels": anchor_label, "filepath": img_path}


class StanfordDogsDataset(Dataset):
    def __init__(self, root_dir, split="test", processor=None):
        self.root_dir = root_dir
        self.processor = processor
        self.images_dir = os.path.join(self.root_dir, "Images")
        if not os.path.exists(self.images_dir) and os.path.exists(os.path.join(self.root_dir, "images", "Images")):
            self.images_dir = os.path.join(self.root_dir, "images", "Images")
            
        self.annotation_dir = os.path.join(self.root_dir, "Annotation")
        if not os.path.exists(self.annotation_dir) and os.path.exists(os.path.join(self.root_dir, "annotations", "Annotation")):
            self.annotation_dir = os.path.join(self.root_dir, "annotations", "Annotation")

        mat_file = "train_list.mat" if split == "train" else "test_list.mat"
        mat = scipy.io.loadmat(os.path.join(self.root_dir, mat_file))

        self.file_list = [f[0][0] for f in mat['file_list']]
        self.labels = [l[0] - 1 for l in mat['labels']]
        
        self.bboxes = []
        for i in range(len(self.file_list)):
            filename = self.file_list[i]
            xml_path = os.path.join(self.annotation_dir, filename.split('.')[0])
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                bndbox = root.find('object').find('bndbox')
                bbox = (int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                        int(bndbox.find('xmax').text), int(bndbox.find('ymax').text))
            except:
                bbox = None
            self.bboxes.append(bbox)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        anchor_label = self.labels[idx]
        filename = self.file_list[idx]
        img_path = os.path.join(self.images_dir, filename)
        
        try:
            image = Image.open(img_path).convert("RGB")
            bbox = self.bboxes[idx]
            if bbox:
                image = image.crop(bbox)
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.processor:
            pixel_values = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        else:
            from torchvision.transforms import Compose, Resize, ToTensor, Normalize
            tf = Compose([Resize((224, 224)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            pixel_values = tf(image)
            
        return {"pixel_values": pixel_values, "labels": anchor_label, "filepath": img_path}


class CIFAR100Dataset(Dataset):
    def __init__(self, split="test", processor=None):
        from datasets import load_dataset
        self.processor = processor
        self.ds = load_dataset("cifar100", split=split)
        
        # Save CIFAR images to disk so we have a filepath for the web interface
        self.img_dir = f"CIFAR100_Images/{split}"
        os.makedirs(self.img_dir, exist_ok=True)
        
    def __len__(self):
        return len(self.ds)
        
    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item['img'].convert("RGB")
        anchor_label = item['fine_label']
        
        filename = f"{idx:05d}_class{anchor_label}.png"
        filepath = os.path.join(self.img_dir, filename)
        if not os.path.exists(filepath):
            img.save(filepath)
        
        if self.processor:
            pixel_values = self.processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
        else:
            from torchvision.transforms import Compose, Resize, ToTensor, Normalize
            tf = Compose([Resize((224, 224)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            pixel_values = tf(img)
            
        return {"pixel_values": pixel_values, "labels": anchor_label, "filepath": filepath}

# ==============================================================================
# MAIN EXTRACTION ENGINE (ĐÃ FIX CHUẨN ĐƯỜNG DẪN VÀO THƯ MỤC MODELS)
# ==============================================================================

MODELS_TO_BUILD = [
    {"model_dir": "models/model_dogs_stanford_final", "dataset": "Stanford_Dogs", "data_root": "Stanford_dogs", "is_vit": True, "base_name": "model_dogs_stanford_final"},
    {"model_dir": "models/model_dogs_resnet50_baseline", "dataset": "Stanford_Dogs", "data_root": "Stanford_dogs", "is_vit": False, "base_name": "model_dogs_resnet50_baseline"},
]

def build_database(model_info):
    model_dir = model_info["model_dir"]
    dataset_name = model_info["dataset"]
    data_root = model_info["data_root"]
    is_vit = model_info["is_vit"]
    base_name = model_info["base_name"]
    
    print(f"\n" + "="*60)
    print(f"[INFO] BẮT ĐẦU XÂY DỰNG DATABASE CHO: {base_name}")
    print(f"="*60)
    
    if not os.path.exists(model_dir):
        print(f"[ERROR] Không tìm thấy thư mục model: {model_dir}. Bỏ qua!")
        return

    # Tự động tạo và lưu vào models/databases để app_pro.py đọc được
    db_dir = os.path.join("models", "databases")
    os.makedirs(db_dir, exist_ok=True)
    index_path = os.path.join(db_dir, f"{base_name}_faiss.index")
    csv_path = os.path.join(db_dir, f"{base_name}_info.csv")
    
    if os.path.exists(index_path) and os.path.exists(csv_path):
        print(f"[INFO] BỎ QUA {base_name} VÌ ĐÃ TRÍCH XUẤT XONG (Đã tìm thấy index và csv).")
        return

    # 1. Load Processor & Model
    print(f"[1/4] Loading model and processor...")
    if is_vit:
        try:
            processor = ViTImageProcessor.from_pretrained(model_dir)
        except Exception:
            processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        model = ViTModel.from_pretrained(model_dir, ignore_mismatched_sizes=True)
    else:
        try:
            processor = AutoImageProcessor.from_pretrained(model_dir)
        except Exception:
            processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        model = ResNetModel.from_pretrained(model_dir, ignore_mismatched_sizes=True)
    
    model.to(DEVICE)
    model.eval()
    print(f"      -> Thành công!")

    # 2. Load Dataset
    print(f"[2/4] Loading Dataset ({dataset_name})...")
    if dataset_name == "CUB200":
        dataset = CUB200Dataset(data_root, split="test", processor=processor)
    elif dataset_name == "Stanford_Dogs":
        dataset = StanfordDogsDataset(data_root, split="test", processor=processor)
    elif dataset_name == "CIFAR-100":
        dataset = CIFAR100Dataset(split="test", processor=processor)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    print(f"      -> Tổng số ảnh trong Database: {len(dataset)}")

    # 3. Extract Features
    print(f"[3/4] Trích xuất Vector (Offline Feature Extraction)...")
    feature_vectors = []
    database_records = []
    faiss_id_counter = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting", unit="batch"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].numpy()
            filepaths = batch["filepath"]
            
            outputs = model(pixel_values=pixel_values)
            
            if is_vit:
                embeddings = outputs.last_hidden_state[:, 0, :]
            else:
                embeddings = outputs.pooler_output.flatten(1)
                
            # Chuẩn hóa L2 (L2 Normalization) để tương thích Cosine Similarity
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            embeddings_np = embeddings.cpu().numpy()
            
            feature_vectors.extend(embeddings_np)
            
            for i in range(len(labels)):
                database_records.append({
                    "faiss_id": faiss_id_counter,
                    "filepath": str(filepaths[i]).replace("\\", "/"),
                    "class_id": labels[i]
                })
                faiss_id_counter += 1

    feature_vectors = np.array(feature_vectors).astype('float32')

    # 4. Build FAISS Index and Save
    print(f"[4/4] Khởi tạo FAISS Index và Lưu trữ...")
    
    # FAISS Index
    dim = feature_vectors.shape[1]
    index = faiss.IndexFlatIP(dim) # Inner Product
    index.add(feature_vectors)
    
    faiss.write_index(index, index_path)
    
    # CSV Metadata
    import pandas as pd
    df = pd.DataFrame(database_records)
    df.to_csv(csv_path, index=False)
    
    print(f"      -> [DONE] Luu Index FAISS tai: {index_path}")
    print(f"      -> [DONE] Luu CSV Metadata tai: {csv_path}")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    print("[START] KHOI DONG HE THONG TRICH XUAT DAC TRUNG OFFLINE (CBIR)")
    for info in MODELS_TO_BUILD:
        build_database(info)
    print("\n[DONE] HOAN TAT TRICH XUAT CHO TAT CA CAC MO HINH VA DATASET!")
