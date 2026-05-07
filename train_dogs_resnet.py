"""
MODULE: SCIENTIFIC BENCHMARK - DOGS (RESNET-50)
MODE: TRIPLET LOSS + BOUNDING BOX CROP + CBIR REPORT GENERATION
"""

import os
import tarfile
import scipy.io
import xml.etree.ElementTree as ET
from urllib.request import urlretrieve
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoImageProcessor, ResNetModel, TrainingArguments, Trainer,
    DefaultDataCollator, EarlyStoppingCallback
)
from sklearn.metrics.pairwise import cosine_similarity

def plot_cbir_metrics(acc, prec, rec, map_score, output_dir, title):
    print(f"[REPORT] Generating CBIR Final Metrics Chart...")
    metrics = ['Accuracy@5', 'Precision@5', 'Recall@5', 'mAP@5']
    values = [acc, prec, rec, map_score]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, values, color=colors)
    plt.title(title)
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 110)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "cbir_metrics_chart.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"✅ Biểu đồ đánh giá CBIR đã được lưu tại: {chart_path}")

# --- CONFIGURATION ---
DATA_ROOT = "/kaggle/working/Stanford_Dogs_Data"
OUTPUT_DIR = "/kaggle/working/"
MODEL_NAME = "microsoft/resnet-50"
# Tối ưu cho Kaggle (30GB RAM & 2x T4 GPUs)
BATCH_SIZE = 64 # Mỗi thanh T4 sẽ gánh 64 ảnh. 2 thanh xử lý tổng cộng 128 ảnh/lượt
GRADIENT_ACCUMULATION_STEPS = 1 # Không cần dồn dốc (Accumulation) vì Batch đã cực lớn (128)
EPOCHS = 40 # Cứ để 40 vì thuật toán Early Stopping sẽ tự động phanh lại khi điểm chạm trần
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# --- DATASET CLASS VỚI LOGIC CẮT BOUNDING BOX VÀ TRIPLET ---
class StanfordDogsDataset(Dataset):
    def __init__(self, root_dir, split="train", processor=None, mode="triplet"):
        self.root_dir = root_dir
        self.split = split
        self.processor = processor
        self.mode = mode
        self.images_dir = os.path.join(self.root_dir, "Images")
        self.annotation_dir = os.path.join(self.root_dir, "Annotation")

        self._check_and_download()
        
        print(f"[INFO] Loading {split} metadata from .mat files...")
        mat_file = "train_list.mat" if split == "train" else "test_list.mat"
        mat = scipy.io.loadmat(os.path.join(self.root_dir, mat_file))

        self.file_list = [f[0][0] for f in mat['file_list']]
        self.labels = [l[0] - 1 for l in mat['labels']]
        self.classes = sorted(list(set([
            f.split('/')[0].split('-')[1] for f in self.file_list
        ])))

        if self.mode == "triplet":
            print(f"--> [INFO] Indexing Triplet Pairs for {split}...")
            self.label_to_indices = defaultdict(list)
            for i, label in enumerate(self.labels):
                self.label_to_indices[label].append(i)
            self.unique_labels = list(self.label_to_indices.keys())

        # [TỐI ƯU HÓA RAM CHỐNG TRÀN EPOCH 6]
        # Thay vì đọc file XML hàng trăm ngàn lần mỗi Epoch gây nghẽn RAM (Page Cache Leak)
        # Ta nạp sẵn toàn bộ Tọa độ Bounding Box vào mảng Array ngay từ đầu!
        print(f"--> [INFO] Pre-parsing XML Bounding Boxes to prevent RAM Leak...")
        self.bboxes = []
        for i in range(len(self.file_list)):
            self.bboxes.append(self._parse_bbox_from_file(i))

        # Khởi tạo bộ công cụ Biến đổi ảnh (Nhanh và không leak RAM bằng AutoImageProcessor)
        image_mean = self.processor.image_mean if self.processor else [0.485, 0.456, 0.406]
        image_std = self.processor.image_std if self.processor else [0.229, 0.224, 0.225]
        
        from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation, ColorJitter
        
        self.train_tf = Compose([
            Resize((224, 224)),
            RandomHorizontalFlip(p=0.5),
            RandomRotation(15),
            ColorJitter(brightness=0.1, contrast=0.1),
            ToTensor(),
            Normalize(mean=image_mean, std=image_std)
        ])
        
        self.val_tf = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=image_mean, std=image_std)
        ])

    def _check_and_download(self):
    # Nếu data đã có sẵn (Kaggle mount) thì bỏ qua hoàn toàn
     if (os.path.exists(os.path.join(self.root_dir, "Images")) and
        os.path.exists(os.path.join(self.root_dir, "Annotation")) and
        os.path.exists(os.path.join(self.root_dir, "train_list.mat"))):
        print(f"[INFO] Dataset found at {self.root_dir}, skipping download.")
        return

    # Nếu chưa có thì download vào /kaggle/working/ (thư mục ghi được)
     download_dir = "/kaggle/working/Stanford_Dogs_Data"
     os.makedirs(download_dir, exist_ok=True)
     self.root_dir = download_dir  # Redirect sang thư mục ghi được
     self.images_dir = os.path.join(download_dir, "Images")
     self.annotation_dir = os.path.join(download_dir, "Annotation")

     urls = {
        "lists.tar": "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar",
        "annotation.tar": "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar",
        "images.tar": "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
     }
     for filename, url in urls.items():
        path = os.path.join(download_dir, filename)
        extract_folder = (
            "Annotation" if "annotation" in filename
            else ("Images" if "images" in filename else "")
        )
        if (extract_folder and not os.path.exists(
                os.path.join(download_dir, extract_folder))) or \
           (not extract_folder and not os.path.exists(
                os.path.join(download_dir, "train_list.mat"))):
            if not os.path.exists(path):
                print(f"--> Downloading {filename}...")
                urlretrieve(url, path)
            print(f"--> Extracting {filename}...")
            with tarfile.open(path, "r") as tar:
                tar.extractall(download_dir)

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
            bbox = self.bboxes[idx] # Tải từ Cache thay vì đọc XML
            if bbox:
                image = image.crop(bbox)
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        return image

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        anchor_label = self.labels[idx]

        if self.mode == "triplet":
            anchor_img = self._load_image(idx)
            pos_idx = random.choice(self.label_to_indices[anchor_label])
            pos_img = self._load_image(pos_idx)

            neg_label = random.choice(self.unique_labels)
            while neg_label == anchor_label:
                neg_label = random.choice(self.unique_labels)
            neg_idx = random.choice(self.label_to_indices[neg_label])
            neg_img = self._load_image(neg_idx)

            # Thay AutoImageProcessor bằng Torchvision để chặn Leak RAM
            if self.split == "train":
                anchor_tensor = self.train_tf(anchor_img)
                pos_tensor = self.train_tf(pos_img)
                neg_tensor = self.train_tf(neg_img)
            else:
                anchor_tensor = self.val_tf(anchor_img)
                pos_tensor = self.val_tf(pos_img)
                neg_tensor = self.val_tf(neg_img)

            return {
                "anchor_values": anchor_tensor,
                "positive_values": pos_tensor,
                "negative_values": neg_tensor,
                "labels": torch.tensor(anchor_label, dtype=torch.long)
            }
        else:
            img = self._load_image(idx)
            pixel_values = self.val_tf(img)
            return {
                "pixel_values": pixel_values,
                "labels": torch.tensor(anchor_label, dtype=torch.long)
            }


# ------------------------------------------------------------------------------
# [BUG FIX #1] ĐỊNH NGHĨA triplet_collate_fn — hàm này bị thiếu hoàn toàn
# Không có hàm này thì Trainer sẽ báo NameError ngay khi khởi tạo
# ------------------------------------------------------------------------------
def triplet_collate_fn(batch):
    anchor_values = torch.stack([item["anchor_values"] for item in batch])
    positive_values = torch.stack([item["positive_values"] for item in batch])
    negative_values = torch.stack([item["negative_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {
        "anchor_values": anchor_values,
        "positive_values": positive_values,
        "negative_values": negative_values,
        "labels": labels,
    }


# ------------------------------------------------------------------------------
# CLASS CUSTOM TRAINER VỚI RESNET DIMENSIONALITY
# ------------------------------------------------------------------------------
class ResNetTripletTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        anchor_imgs = inputs.get("anchor_values")
        positive_imgs = inputs.get("positive_values")
        negative_imgs = inputs.get("negative_values")

        anchor_vec = model(pixel_values=anchor_imgs).pooler_output.flatten(1)
        positive_vec = model(pixel_values=positive_imgs).pooler_output.flatten(1)
        negative_vec = model(pixel_values=negative_imgs).pooler_output.flatten(1)

        anchor_vec = nn.functional.normalize(anchor_vec, p=2, dim=1)
        positive_vec = nn.functional.normalize(positive_vec, p=2, dim=1)
        negative_vec = nn.functional.normalize(negative_vec, p=2, dim=1)

        triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
        loss = triplet_loss_fn(anchor_vec, positive_vec, negative_vec)

        return (loss, None) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        import gc
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, return_outputs=False)
        
        # Ngắt đồ thị và ép dọn dẹp RAM (Chống tràn 30GB Kaggle)
        loss = loss.detach()
        del inputs
        gc.collect()
        
        return (loss, None, None)


# ------------------------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------------------------
def plot_training_history(log_history, output_dir):
    print(f"[REPORT] Generating training charts...")
    loss_steps, loss_values = [], []
    eval_steps, eval_loss_values = [], []

    for entry in log_history:
        if 'loss' in entry:
            loss_steps.append(entry['step'])
            loss_values.append(entry['loss'])
        if 'eval_loss' in entry:
            eval_steps.append(entry['step'])
            eval_loss_values.append(entry['eval_loss'])

    plt.figure(figsize=(10, 5))
    plt.plot(loss_steps, loss_values, label='Training Loss', color='tab:red')
    if len(eval_steps) > 0:
        plt.plot(eval_steps, eval_loss_values, label='Eval Loss', color='tab:blue')

    plt.title('Training Loss Tracking')
    plt.xlabel('Steps')
    plt.ylabel('Triplet Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "training_charts.png")
    plt.savefig(chart_path)
    print(f"[REPORT] Charts saved to: {chart_path}")
    plt.close()


# --- MAIN ---
def main():
    print(f"[INIT] TRAINING DOGS RESNET-50 (CBIR) | BBOX CROP: ENABLED | EPOCHS: {EPOCHS}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    train_ds = StanfordDogsDataset(DATA_ROOT, split="train", processor=processor, mode="triplet")
    test_ds_triplet = StanfordDogsDataset(DATA_ROOT, split="test", processor=processor, mode="triplet")

    id2label = {str(i): c for i, c in enumerate(train_ds.classes)}
    label2id = {c: str(i) for i, c in enumerate(train_ds.classes)}

    model = ResNetModel.from_pretrained(
        MODEL_NAME, num_labels=len(train_ds.classes),
        id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    )

    print("[INFO] Đang đóng băng (Freezing) Embedder và Stage 1-2 của ResNet-50...")
    for param in model.embedder.parameters():
        param.requires_grad = False
    for stage in model.encoder.stages[:2]:
        for param in stage.parameters():
            param.requires_grad = False

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=1,
        fp16=(DEVICE == "cuda"),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        dataloader_pin_memory=False,
        dataloader_num_workers=4, # Đã fix lỗi RAM nên bật max 4 lõi CPU để data cho 2 GPU
        remove_unused_columns=False,
    )

    trainer = ResNetTripletTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds_triplet,
        data_collator=triplet_collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("\n" + "=" * 60)
    print("--> STARTING STANFORD DOGS RESNET TRIPLET LOSS TRAINING LOOP")
    print("=" * 60 + "\n")

    torch.cuda.empty_cache()
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    plot_training_history(trainer.state.log_history, OUTPUT_DIR)

    # =====================================================================
    # ĐÁNH GIÁ CBIR METRICS NGAY SAU KHI TRAIN XONG
    # =====================================================================
    print("\n" + "=" * 60)
    print("--> BẮT ĐẦU ĐÁNH GIÁ ĐỘ ĐO CBIR (mAP, Recall@5, Precision@5)")
    print("=" * 60)

    eval_dataset = StanfordDogsDataset(DATA_ROOT, split="test", processor=processor, mode="eval")
    test_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

    model.to(DEVICE)
    model.eval()
    feature_vectors = []
    labels_list = []

    print("Đang trích xuất vector từ tập Test...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting Features"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].numpy()

            # [BUG FIX #3] model(pixel_values) → model(pixel_values=pixel_values)
            outputs = model(pixel_values=pixel_values)
            embeddings = outputs.pooler_output.flatten(1)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            feature_vectors.extend(embeddings.cpu().numpy())
            labels_list.extend(labels)

    feature_vectors = np.array(feature_vectors)
    labels_array = np.array(labels_list)

    print("Đang tính toán ma trận Cosine và các độ đo...")
    TOP_K = 5
    similarity_matrix = cosine_similarity(feature_vectors, feature_vectors)

    num_samples = len(labels_array)
    accuracy_count = 0
    precision_scores = []
    recall_scores = []
    average_precisions = []

    for i in range(num_samples):
        query_label = labels_array[i]
        similarities = similarity_matrix[i]

        ranked_indices = np.argsort(similarities)[::-1]
        ranked_indices = ranked_indices[ranked_indices != i]

        top_k_indices = ranked_indices[:TOP_K]
        retrieved_labels = labels_array[top_k_indices]

        relevant_mask = (retrieved_labels == query_label)
        num_relevant_retrieved = np.sum(relevant_mask)

        if num_relevant_retrieved > 0:
            accuracy_count += 1

        precision_scores.append(num_relevant_retrieved / TOP_K)

        total_relevant = np.sum(labels_array == query_label) - 1
        recall_scores.append(
            num_relevant_retrieved / total_relevant if total_relevant > 0 else 0
        )

        precisions_at_hits = []
        hit_count = 0
        for rank, idx in enumerate(top_k_indices, start=1):
            if labels_array[idx] == query_label:
                hit_count += 1
                precisions_at_hits.append(hit_count / rank)
        average_precisions.append(
            np.mean(precisions_at_hits) if precisions_at_hits else 0
        )

    acc_final = accuracy_count / num_samples * 100
    prec_final = np.mean(precision_scores) * 100
    rec_final = np.mean(recall_scores) * 100
    map_final = np.mean(average_precisions) * 100

    print("\n" + "=" * 60)
    print("🚀 BÁO CÁO KẾT QUẢ HUẤN LUYỆN CBIR STANFORD DOGS RESNET")
    print("=" * 60)
    print(f"Accuracy@{TOP_K}:  {acc_final:.2f}%")
    print(f"Precision@{TOP_K}: {prec_final:.2f}%")
    print(f"Recall@{TOP_K}:    {rec_final:.2f}%")
    print(f"mAP@{TOP_K}:       {map_final:.2f}%")
    print("=" * 60)

    plot_cbir_metrics(
        acc_final, prec_final, rec_final, map_final, 
        OUTPUT_DIR, "Final CBIR Performance on Stanford Dogs (ResNet-50)"
    )


if __name__ == "__main__":
    main()