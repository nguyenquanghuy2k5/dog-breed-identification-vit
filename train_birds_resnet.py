"""
MODULE: SCIENTIFIC BENCHMARK - BIRDS (RESNET-50)
MODE: TRIPLET LOSS + BOUNDING BOX CROP + CBIR REPORT GENERATION
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoImageProcessor, ResNetModel, TrainingArguments, Trainer, DefaultDataCollator, EarlyStoppingCallback
)
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter, Compose

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
# >>> [ĐỒNG BỘ: KAGGLE PATH CỦA CUB-200]
DATA_ROOT = "/kaggle/input/datasets/thinhtran18/cub-200-2011"
OUTPUT_DIR = "./model_birds_resnet50_baseline"
MODEL_NAME = "microsoft/resnet-50"

# ResNet tốn rấy ít VRAM (nhẹ hơn ViT), có thể chạy BATCH 16 thoải mái trên Colab T4
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4

EPOCHS = 40  
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

ImageFile.LOAD_TRUNCATED_IMAGES = True 
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- DATASET CLASS VỚI LOGIC CẮT BOUNDING BOX VÀ TRIPLET ---
class CUB200Dataset(Dataset):
    def __init__(self, root_dir, split="train", processor=None, mode="triplet"):
        self.root_dir = root_dir
        self.split = split
        self.images_dir = os.path.join(root_dir, "images")
        self.processor = processor
        self.mode = mode
        
        # Check files
        for f in ['images.txt', 'train_test_split.txt', 'image_class_labels.txt', 'bounding_boxes.txt', 'classes.txt']:
            if not os.path.exists(os.path.join(root_dir, f)):
                raise FileNotFoundError(f"[CRITICAL] Thiếu file '{f}' trong {root_dir}")

        # 1. Load Metadata
        df_imgs = pd.read_csv(os.path.join(root_dir, 'images.txt'), sep=' ', names=['id', 'filepath'])
        df_split = pd.read_csv(os.path.join(root_dir, 'train_test_split.txt'), sep=' ', names=['id', 'is_train'])
        df_labels = pd.read_csv(os.path.join(root_dir, 'image_class_labels.txt'), sep=' ', names=['id', 'class_id'])
        df_bbox = pd.read_csv(os.path.join(root_dir, 'bounding_boxes.txt'), sep=' ', names=['id', 'x', 'y', 'w', 'h'])
        df_classes = pd.read_csv(os.path.join(root_dir, 'classes.txt'), sep=' ', names=['id', 'class_name'])

        # 2. Merge Data
        df = pd.merge(df_imgs, df_split, on='id')
        df = pd.merge(df, df_labels, on='id')
        df = pd.merge(df, df_bbox, on='id') # Thêm thông tin BBox

        target = 1 if split == "train" else 0
        self.data = df[df['is_train'] == target].reset_index(drop=True)
        self.classes = df_classes['class_name'].tolist()
        
        # CUB Label bắt đầu từ 1, ta trừ 1 để đưa về index 0-based
        self.labels = [int(x) - 1 for x in self.data['class_id'].tolist()]
        
        print(f"[DATA] {split.upper()} Set: {len(self.data)} images (Official Split + BBox | Mode: {mode}).")

        # Prepare Triplet Indices for "triplet" mode lookup
        if self.mode == "triplet":
            self.label_to_indices = defaultdict(list)
            for i, label in enumerate(self.labels):
                self.label_to_indices[label].append(i)
            self.unique_labels = list(self.label_to_indices.keys())
            
            # >>> [ĐÃ ĐỒNG BỘ: CHỐNG OVERFIT TRONG TRIPLET BẰNG AUGMENTATION]
            self.augment = Compose([
                RandomHorizontalFlip(p=0.5),
                RandomRotation(15),
                ColorJitter(brightness=0.2, contrast=0.2)
            ])

        from torchvision.transforms import ToTensor, Normalize, Resize
        self.base_transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_image(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.images_dir, row['filepath'])
        try:
            image = Image.open(img_path).convert("RGB")
            # --- CRITICAL: CẮT ẢNH THEO BOUNDING BOX ---
            # Ép kiểu int() khắc phục triệt để lỗi tọa độ float trên Kaggle trả về ảnh đen
            x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
            image = image.crop((x, y, x + w, y + h))
        except Exception as e:
            print(f"[WARNING] Lỗi Crop ảnh tại dòng {idx}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        return image

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        anchor_label = self.labels[idx]
        
        if self.mode == "triplet":
            # 1. ANCHOR
            anchor_img = self._load_image(idx)
            
            # 2. POSITIVE
            pos_idx = random.choice(self.label_to_indices[anchor_label])
            pos_img = self._load_image(pos_idx)
            
            # 3. NEGATIVE
            neg_label = random.choice(self.unique_labels)
            while neg_label == anchor_label:
                neg_label = random.choice(self.unique_labels)
            neg_idx = random.choice(self.label_to_indices[neg_label])
            neg_img = self._load_image(neg_idx)
            
            # >>> CHIẾU CÁC HIỆU ỨNG BIẾN ĐỔI (Data Augmentation) LÊN CÁC KHUNG HÌNH
            # QUAN TRỌNG: Chỉ bật Augment khi đang học (Train). Tắt nó khi Validation (Test) để đồ thị Eval_Loss ổn định!
            if self.split == "train":
                anchor_img = self.augment(anchor_img)
                pos_img = self.augment(pos_img)
                neg_img = self.augment(neg_img)
            
            anchor_tensor = self.base_transform(anchor_img)
            pos_tensor = self.base_transform(pos_img)
            neg_tensor = self.base_transform(neg_img)

            return {
                "anchor_values": anchor_tensor,
                "positive_values": pos_tensor,
                "negative_values": neg_tensor,
                "labels": torch.tensor(anchor_label, dtype=torch.long)
            }
        else:
            img = self._load_image(idx)
            pixel_values = self.base_transform(img)
            return {
                "pixel_values": pixel_values,
                "labels": torch.tensor(anchor_label, dtype=torch.long)
            }

# ------------------------------------------------------------------------------
# CLASS CUSTOM TRAINER VỚI RESNET DIMENSIONALITY 
# ------------------------------------------------------------------------------
from transformers import TrainerCallback
import gc

class MemoryCleanupCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # Dọn rác System RAM (Phòng chống tràn RAM 30GB Kaggle)
        gc.collect()
        # Dọn rác VRAM GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n[INFO] Đã dọn dẹp System RAM & VRAM thành công ở cuối Epoch.")

class ResNetTripletTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        anchor_imgs = inputs.get("anchor_values")
        positive_imgs = inputs.get("positive_values")
        negative_imgs = inputs.get("negative_values")

        # TỐI ƯU HÓA CỰC HẠN: Gộp chung 3 mảng ảnh thành 1 Batch duy nhất
        # Thay vì chạy model 3 lần (tạo 3 đồ thị PyTorch cồng kềnh), ta chỉ chạy 1 lần duy nhất!
        combined_imgs = torch.cat([anchor_imgs, positive_imgs, negative_imgs], dim=0)
        
        # Đẩy qua ResNet lấy mảng POOLER_OUTPUT (Kích thước: 3*batch x 2048 x 1 x 1)
        outputs = model(pixel_values=combined_imgs)
        embeddings = outputs.pooler_output.flatten(1)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        # Cắt trả lại thành 3 phần (Anchor, Positive, Negative)
        batch_size = anchor_imgs.size(0)
        anchor_vec = embeddings[:batch_size]
        positive_vec = embeddings[batch_size:2*batch_size]
        negative_vec = embeddings[2*batch_size:]

        # Tính Triplet Loss
        triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
        loss = triplet_loss_fn(anchor_vec, positive_vec, negative_vec)

        return (loss, None) if return_outputs else loss

    # =========================================================================
    # [BẢN VÁ LỖI EVALUATION] BẮT BUỘC PHẢI CÓ ĐOẠN NÀY ĐỂ KHÔNG BỊ CRASH
    # =========================================================================
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
    print(f"[INIT] TRAINING BIRDS RESNET-50 (CBIR) | BBOX CROP: ENABLED | EPOCHS: {EPOCHS}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    
    try:
        train_ds = CUB200Dataset(DATA_ROOT, split="train", processor=processor, mode="triplet")
        test_ds_triplet = CUB200Dataset(DATA_ROOT, split="test", processor=processor, mode="triplet")
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # Clean class names
    labels_clean = [c.split('.')[-1].replace('_', ' ') for c in train_ds.classes]
    id2label = {str(i): c for i, c in enumerate(labels_clean)}
    label2id = {c: str(i) for i, c in enumerate(labels_clean)}
    
    model = ResNetModel.from_pretrained(
        MODEL_NAME, num_labels=len(labels_clean),
        id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    )
    
    # =========================================================================
    # FREEZING LAYER (CHỐNG OVERFITTING & MAX SPEED COLAB)
    # =========================================================================
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
        fp16=(DEVICE=="cuda"),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        logging_steps=10
    )
    
    trainer = ResNetTripletTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds_triplet,
        data_collator=DefaultDataCollator(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), MemoryCleanupCallback()]
    )
    
    print("\n" + "="*60)
    print("--> STARTING CUB-200 RESNET TRIPLET LOSS TRAINING LOOP")
    print("="*60 + "\n")
    
    torch.cuda.empty_cache()
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    plot_training_history(trainer.state.log_history, OUTPUT_DIR)
    
    # =====================================================================
    # ĐÁNH GIÁ CBIR METRICS NGAY SAU KHI TRAIN XONG 
    # =====================================================================
    print("\n" + "="*60)
    print("--> BẮT ĐẦU ĐÁNH GIÁ ĐỘ ĐO CBIR (mAP, Recall@5, Precision@5)")
    print("="*60)
    
    eval_dataset = CUB200Dataset(DATA_ROOT, split="test", processor=processor, mode="eval")
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
            
            outputs = model(pixel_values)
            # DÙNG POOLER OUTPUT CHUẨN CỦA RESNET
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
        recall_scores.append(num_relevant_retrieved / total_relevant if total_relevant > 0 else 0)
        
        precisions_at_hits = []
        hit_count = 0
        for rank, idx in enumerate(top_k_indices, start=1):
            if labels_array[idx] == query_label:
                hit_count += 1
                precisions_at_hits.append(hit_count / rank)
        average_precisions.append(np.mean(precisions_at_hits) if precisions_at_hits else 0)

    acc_final = accuracy_count / num_samples * 100
    prec_final = np.mean(precision_scores) * 100
    rec_final = np.mean(recall_scores) * 100
    map_final = np.mean(average_precisions) * 100

    print("\n" + "="*60)
    print("🚀 BÁO CÁO KẾT QUẢ HUẤN LUYỆN CBIR CUB-200 RESNET")
    print("="*60)
    print(f"Accuracy@{TOP_K}:  {acc_final:.2f}%")
    print(f"Precision@{TOP_K}: {prec_final:.2f}%")
    print(f"Recall@{TOP_K}:    {rec_final:.2f}%")
    print(f"mAP@{TOP_K}:       {map_final:.2f}%")
    print("="*60)

    plot_cbir_metrics(
        acc_final, prec_final, rec_final, map_final, 
        OUTPUT_DIR, "Final CBIR Performance on CUB-200 (ResNet-50)"
    )

if __name__ == "__main__":
    main()