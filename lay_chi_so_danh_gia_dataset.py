import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scipy.io
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoImageProcessor, ResNetModel, ViTModel, ViTImageProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import sys
import io

# Force UTF-8 encoding for Windows terminals to avoid UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# DATASET CLASSES (Local Versions)
# ==============================================================================

class CUB200Dataset(Dataset):
    def __init__(self, root_dir, split="test", processor=None):
        self.root_dir = root_dir
        self.split = split
        self.images_dir = os.path.join(root_dir, "images")
        self.processor = processor
        
        df_imgs = pd.read_csv(os.path.join(root_dir, 'images.txt'), sep=' ', names=['id', 'filepath'])
        df_split = pd.read_csv(os.path.join(root_dir, 'train_test_split.txt'), sep=' ', names=['id', 'is_train'])
        df_labels = pd.read_csv(os.path.join(root_dir, 'image_class_labels.txt'), sep=' ', names=['id', 'class_id'])
        df_bbox = pd.read_csv(os.path.join(root_dir, 'bounding_boxes.txt'), sep=' ', names=['id', 'x', 'y', 'w', 'h'])
        df_classes = pd.read_csv(os.path.join(root_dir, 'classes.txt'), sep=' ', names=['id', 'class_name'])

        df = pd.merge(df_imgs, df_split, on='id')
        df = pd.merge(df, df_labels, on='id')
        df = pd.merge(df, df_bbox, on='id')

        target = 1 if split == "train" else 0
        self.data = df[df['is_train'] == target].reset_index(drop=True)
        self.classes = df_classes['class_name'].tolist()
        self.labels = [int(x) - 1 for x in self.data['class_id'].tolist()]

    def _load_image(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.images_dir, row['filepath'])
        try:
            image = Image.open(img_path).convert("RGB")
            x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
            image = image.crop((x, y, x + w, y + h))
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        return image

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self._load_image(idx)
        anchor_label = self.labels[idx]
        pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(anchor_label, dtype=torch.long)
        }

class StanfordDogsDataset(Dataset):
    def __init__(self, root_dir, split="test", processor=None):
        self.root_dir = root_dir
        self.split = split
        self.processor = processor
        self.images_dir = os.path.join(self.root_dir, "Images")
        if not os.path.exists(self.images_dir) and os.path.exists(os.path.join(self.root_dir, "images", "Images")):
            self.images_dir = os.path.join(self.root_dir, "images", "Images")
            
        self.annotation_dir = os.path.join(self.root_dir, "Annotation")
        if not os.path.exists(self.annotation_dir) and os.path.exists(os.path.join(self.root_dir, "annotations", "Annotation")):
            self.annotation_dir = os.path.join(self.root_dir, "annotations", "Annotation")

        mat_file = "train_list.mat" if split == "train" else "test_list.mat"
        mat_path = os.path.join(self.root_dir, mat_file)
        
        if not os.path.exists(mat_path):
            import urllib.request
            import tarfile
            print(f"\n[INFO] Không tìm thấy '{mat_file}', đang tự động tải từ Stanford Server...")
            tar_path = os.path.join(self.root_dir, "lists.tar")
            if not os.path.exists(tar_path):
                urllib.request.urlretrieve("http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar", tar_path)
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(self.root_dir)
            print("[INFO] Đã tải và giải nén xong!\n")

        mat = scipy.io.loadmat(mat_path)

        self.file_list = [f[0][0] for f in mat['file_list']]
        self.labels = [l[0] - 1 for l in mat['labels']]
        
        self.bboxes = []
        for i in range(len(self.file_list)):
            self.bboxes.append(self._parse_bbox_from_file(i))

    def _parse_bbox_from_file(self, idx):
        filename = self.file_list[idx]
        xml_path = os.path.join(self.annotation_dir, filename.split('.')[0])
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            bndbox = root.find('object').find('bndbox')
            return (
                int(bndbox.find('xmin').text),
                int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text),
                int(bndbox.find('ymax').text)
            )
        except:
            return None

    def _load_image(self, idx):
        img_path = os.path.join(self.images_dir, self.file_list[idx])
        try:
            image = Image.open(img_path).convert("RGB")
            bbox = self.bboxes[idx]
            if bbox:
                image = image.crop(bbox)
        except Exception as e:
            print(f"Lỗi tải ảnh {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        return image

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        anchor_label = self.labels[idx]
        img = self._load_image(idx)
        
        # Torchvision transforms setup specific for Dogs (to mirror train script)
        from torchvision.transforms import Compose, Resize, ToTensor, Normalize
        image_mean = self.processor.image_mean if hasattr(self.processor, "image_mean") else [0.485, 0.456, 0.406]
        image_std = self.processor.image_std if hasattr(self.processor, "image_std") else [0.229, 0.224, 0.225]
        
        val_tf = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=image_mean, std=image_std)
        ])
        
        pixel_values = val_tf(img)
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(anchor_label, dtype=torch.long)
        }

class CIFAR100Dataset(Dataset):
    def __init__(self, split="test", processor=None):
        from datasets import load_dataset
        self.processor = processor
        # CIFAR-100 uses 'train' and 'test' splits
        self.ds = load_dataset("cifar100", split=split)
        
    def __len__(self):
        return len(self.ds)
        
    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item['img'].convert("RGB")
        anchor_label = item['fine_label']
        
        if self.processor:
            pixel_values = self.processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
            return {"pixel_values": pixel_values, "labels": torch.tensor(anchor_label, dtype=torch.long)}
        else:
            from torchvision.transforms import Compose, Resize, ToTensor, Normalize
            image_mean = [0.485, 0.456, 0.406]
            image_std = [0.229, 0.224, 0.225]
            tf = Compose([
                Resize((224, 224)),
                ToTensor(),
                Normalize(mean=image_mean, std=image_std)
            ])
            pixel_values = tf(img)
            return {"pixel_values": pixel_values, "labels": torch.tensor(anchor_label, dtype=torch.long)}

# ==============================================================================
# EVALUATION ENGINE
# ==============================================================================

def evaluate_model(model_dir, dataset_name, data_root, is_vit):
    print(f"\n[{model_dir}] Khởi tạo đánh giá...")
    if not os.path.exists(model_dir):
        print(f"[ERROR] Không tìm thấy thư mục model: {model_dir}")
        return None

    # 1. Load Processor & Model
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

    # 2. Load Dataset
    if dataset_name == "CUB200":
        eval_dataset = CUB200Dataset(data_root, split="test", processor=processor)
    elif dataset_name == "Stanford_Dogs":
        eval_dataset = StanfordDogsDataset(data_root, split="test", processor=processor)
    elif dataset_name == "CIFAR-100":
        eval_dataset = CIFAR100Dataset(split="test", processor=processor)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    test_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    
    feature_vectors = []
    labels_list = []

    # 3. Extract Features
    print(f"[{model_dir}] Đang trích xuất Vector từ {len(eval_dataset)} ảnh Test...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].numpy()
            
            outputs = model(pixel_values=pixel_values)
            
            if is_vit:
                embeddings = outputs.last_hidden_state[:, 0, :]
            else:
                embeddings = outputs.pooler_output.flatten(1)
                
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            feature_vectors.extend(embeddings.cpu().numpy())
            labels_list.extend(labels)

    feature_vectors = np.array(feature_vectors)
    labels_array = np.array(labels_list)

    # 4. Compute Metrics (mAP@5, Recall@5, Precision@5)
    print(f"[{model_dir}] Đang tính toán CBIR Matrix...")
    TOP_K = 5
    similarity_matrix = cosine_similarity(feature_vectors, feature_vectors)

    num_samples = len(labels_array)
    accuracy_count = 0
    precision_scores = []
    recall_scores = []
    average_precisions = []

    for i in tqdm(range(num_samples), desc="Scoring"):
        query_label = labels_array[i]
        similarities = similarity_matrix[i]

        ranked_indices = np.argsort(similarities)[::-1]
        ranked_indices = ranked_indices[ranked_indices != i] # Bỏ qua bản thân nó

        top_k_indices = ranked_indices[:TOP_K]
        retrieved_labels = labels_array[top_k_indices]

        relevant_mask = (retrieved_labels == query_label)
        num_relevant_retrieved = np.sum(relevant_mask)

        if num_relevant_retrieved > 0:
            accuracy_count += 1

        precision_scores.append(num_relevant_retrieved / TOP_K)

        total_relevant = np.sum(labels_array == query_label) - 1
        recall_scores.append(num_relevant_retrieved / total_relevant if total_relevant > 0 else 0)

        precisions_at_hits = []
        hit_count = 0
        for rank, idx in enumerate(top_k_indices, start=1):
            if labels_array[idx] == query_label:
                hit_count += 1
                precisions_at_hits.append(hit_count / rank)
        average_precisions.append(np.mean(precisions_at_hits) if precisions_at_hits else 0)

    acc = (accuracy_count / num_samples) * 100
    prec = np.mean(precision_scores) * 100
    rec = np.mean(recall_scores) * 100
    mAP = np.mean(average_precisions) * 100
    
    return {"Accuracy@5": acc, "Precision@5": prec, "Recall@5": rec, "mAP@5": mAP}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    configs = [
        {"model_dir": "model_birds_cub200_final", "dataset": "CUB200", "data_root": "Cub_200_2011", "is_vit": True},
        {"model_dir": "model_birds_resnet50_baseline", "dataset": "CUB200", "data_root": "Cub_200_2011", "is_vit": False},
        {"model_dir": "model_dogs_stanford_final", "dataset": "Stanford_Dogs", "data_root": "Stanford_dogs", "is_vit": True},
        {"model_dir": "model_dogs_resnet50_baseline", "dataset": "Stanford_Dogs", "data_root": "Stanford_dogs", "is_vit": False},
        {"model_dir": "model_cifar-100_final", "dataset": "CIFAR-100", "data_root": None, "is_vit": True},
        {"model_dir": "cifar100_resnet50", "dataset": "CIFAR-100", "data_root": None, "is_vit": False}
    ]

    results = {}
    for cfg in configs:
        res = evaluate_model(cfg["model_dir"], cfg["dataset"], cfg["data_root"], cfg["is_vit"])
        if res:
            results[cfg["model_dir"]] = res

    print("\n" + "="*80)
    print(" BẢNG TỔNG HỢP CHỈ SỐ ĐÁNH GIÁ CBIR (TEST SET)")
    print("="*80)
    print(f"{'MÔ HÌNH (THƯ MỤC)':<35} | {'Acc@5 (%)':<10} | {'Prec@5 (%)':<10} | {'Rec@5 (%)':<10} | {'mAP@5 (%)':<10}")
    print("-" * 80)
    for model_name, metrics in results.items():
        print(f"{model_name:<35} | {metrics['Accuracy@5']:<10.2f} | {metrics['Precision@5']:<10.2f} | {metrics['Recall@5']:<10.2f} | {metrics['mAP@5']:<10.2f}")
    print("="*80)
    
    # Save to text file
    with open("cbir_evaluation_results.txt", "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(" BẢNG TỔNG HỢP CHỈ SỐ ĐÁNH GIÁ CBIR (TEST SET)\n")
        f.write("="*80 + "\n")
        f.write(f"{'MÔ HÌNH (THƯ MỤC)':<35} | {'Acc@5 (%)':<10} | {'Prec@5 (%)':<10} | {'Rec@5 (%)':<10} | {'mAP@5 (%)':<10}\n")
        f.write("-" * 80 + "\n")
        for model_name, metrics in results.items():
            f.write(f"{model_name:<35} | {metrics['Accuracy@5']:<10.2f} | {metrics['Precision@5']:<10.2f} | {metrics['Recall@5']:<10.2f} | {metrics['mAP@5']:<10.2f}\n")
        f.write("="*80 + "\n")
    print("\n✅ Đã lưu kết quả ra file: cbir_evaluation_results.txt")
