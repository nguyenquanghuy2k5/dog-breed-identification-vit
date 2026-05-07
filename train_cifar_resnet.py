# ==============================================================================
# MODULE: CIFAR-100 BENCHMARK PIPELINE - RESNET-50
# MODE: TRIPLET LOSS + HUGGING FACE DATASETS + CBIR METRICS CHART
# ==============================================================================

import os
import torch
import numpy as np
import torch.nn as nn
from transformers import Trainer
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from datasets import load_dataset
from transformers import (
    AutoImageProcessor, ResNetModel, TrainingArguments, Trainer, DefaultDataCollator, EarlyStoppingCallback
)
from torchvision.transforms import (
    CenterCrop, Compose, Normalize, RandomHorizontalFlip, Resize, ToTensor, RandomRotation, ColorJitter
)
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True  
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# ------------------------------------------------------------------------------
# 1. CẤU HÌNH HỆ THỐNG
# ------------------------------------------------------------------------------
OUTPUT_DIR  = "./model_cifar_resnet50_final"
MODEL_CHECKPOINT = "microsoft/resnet-50"
# Tối ưu cho Kaggle (2x T4 GPUs)
BATCH_SIZE = 32 # Hạ Batch Size xuống 32 (Mỗi T4 gánh 32*3 ảnh Triplet = 96 ảnh) chống tràn VRAM
GRADIENT_ACCUMULATION_STEPS = 2 # Tăng dồn dốc lên 2 để giữ nguyên Effective Batch Size (128)
EPOCHS = 20  
LEARNING_RATE = 2e-5

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"--> [INIT] BẮT ĐẦU KÉO DATASET CIFAR-100 TỪ HUGGING FACE...")
ds = load_dataset("cifar100")

# Trích xuất nhãn (100 loại)
labels = ds["train"].features["fine_label"].names
id2label = {str(i): c for i, c in enumerate(labels)}
label2id = {c: str(i) for i, c in enumerate(labels)}

print(f"--> [INFO] Đã load {len(ds['train'])} ảnh Train và {len(ds['test'])} ảnh Test.")

# ------------------------------------------------------------------------------
# 2. XÂY DỰNG TRIPLET LOSS TRÊN DATASET
# ------------------------------------------------------------------------------
processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)
image_mean = processor.image_mean
image_std = processor.image_std
size = processor.size["shortest_edge"]

# Tích hợp Resize(224) để phóng to ảnh 32x32 của Cifar-100
_train_tf = Compose([
    Resize((size, size)),
    RandomHorizontalFlip(p=0.5), 
    RandomRotation(15), 
    ColorJitter(brightness=0.1, contrast=0.1),
    ToTensor(), 
    Normalize(mean=image_mean, std=image_std)
])

_val_tf = Compose([
    Resize((size, size)),
    ToTensor(), 
    Normalize(mean=image_mean, std=image_std)
])
# [BUG FIX TỐI THƯỢNG - GIẢI QUYẾT TRIỆT ĐỂ TRÀN RAM]
# Thay vì dùng HuggingFace Dataset và PIL Image (gây rò rỉ bộ nhớ C-pointers khi chạy đa luồng),
# Ta sẽ nén 100% ảnh thành mảng Numpy nguyên thủy và dùng class Dataset chuẩn của PyTorch!
print("--> [INFO] Ép kiểu toàn bộ Cifar-100 thành Numpy Arrays để xóa sổ Memory Leak đa luồng...")
train_imgs_np = np.array([np.array(img.convert("RGB")) for img in tqdm(ds["train"]["img"], desc="Numpy Caching Train")])
train_labels = np.array(ds["train"]["fine_label"])

test_imgs_np = np.array([np.array(img.convert("RGB")) for img in tqdm(ds["test"]["img"], desc="Numpy Caching Test")])
test_labels = np.array(ds["test"]["fine_label"])

class PureCifarTripletDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_np, labels_np, transform, is_train=True):
        self.imgs_np = imgs_np
        self.labels_np = labels_np
        self.transform = transform
        self.is_train = is_train
        
        if self.is_train:
            self.label_to_indices = defaultdict(list)
            for i, lbl in enumerate(self.labels_np):
                self.label_to_indices[lbl].append(i)
            self.unique_labels = list(self.label_to_indices.keys())

    def __len__(self):
        return len(self.labels_np)

    def __getitem__(self, idx):
        anchor_img = Image.fromarray(self.imgs_np[idx])
        anchor_label = int(self.labels_np[idx])
        
        if self.is_train:
            pos_idx = random.choice(self.label_to_indices[anchor_label])
            pos_img = Image.fromarray(self.imgs_np[pos_idx])
            
            neg_label = random.choice(self.unique_labels)
            while neg_label == anchor_label:
                neg_label = random.choice(self.unique_labels)
            neg_idx = random.choice(self.label_to_indices[neg_label])
            neg_img = Image.fromarray(self.imgs_np[neg_idx])
            
            return {
                "anchor_values": self.transform(anchor_img),
                "positive_values": self.transform(pos_img),
                "negative_values": self.transform(neg_img),
                "labels": torch.tensor(anchor_label, dtype=torch.long)
            }
        else:
            return {
                "pixel_values": self.transform(anchor_img),
                "labels": torch.tensor(anchor_label, dtype=torch.long)
            }

train_dataset_triplet = PureCifarTripletDataset(train_imgs_np, train_labels, _train_tf, is_train=True)
test_dataset_triplet = PureCifarTripletDataset(test_imgs_np, test_labels, _val_tf, is_train=True) # Eval Triplet loss

# ------------------------------------------------------------------------------
# 3. CẤU HÌNH TRAINING VỚI RESNET-50 (ĐÓNG BĂNG LAYER)
# ------------------------------------------------------------------------------
model = ResNetModel.from_pretrained(
    MODEL_CHECKPOINT, 
    num_labels=len(labels), 
    id2label=id2label, 
    label2id=label2id, 
    ignore_mismatched_sizes=True
)

print("[INFO] Đang đóng băng (Freezing) Embedder và Stage 1-2 của ResNet-50...")
for param in model.embedder.parameters():
    param.requires_grad = False
for stage in model.encoder.stages[:2]:
    for param in stage.parameters():
        param.requires_grad = False

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,            
    learning_rate=LEARNING_RATE,             
    lr_scheduler_type="cosine",     
    per_device_train_batch_size=BATCH_SIZE,  
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,             
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss", 
    greater_is_better=False,           
    remove_unused_columns=False,    
    report_to="none",
    fp16=(DEVICE=="cuda"),        
    dataloader_num_workers=0, # TẮT Worker để triệt tiêu mọi rủi ro copy-on-write của Multiprocessing
    logging_steps=195 # ~0.5 Epoch
)

import gc
from transformers import TrainerCallback

class MemoryCleanupCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # Dọn rác System RAM (Phòng chống tràn RAM 30GB Kaggle)
        gc.collect()
        # Dọn rác VRAM GPU
        torch.cuda.empty_cache()
        print("\n[INFO] 🧹 Đã cưỡng bức dọn dẹp System RAM & VRAM thành công ở cuối Epoch.")

# THIẾT LẬP COLLATE_FN DÀNH CHO PURE DATASET
def triplet_collate_fn(batch):
    return {
        "anchor_values": torch.stack([item["anchor_values"] for item in batch]),
        "positive_values": torch.stack([item["positive_values"] for item in batch]),
        "negative_values": torch.stack([item["negative_values"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch])
    }

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

        triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
        loss = triplet_loss_fn(anchor_vec, positive_vec, negative_vec)
        return (loss, None) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, return_outputs=False)
        return (loss, None, None)

trainer = ResNetTripletTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_triplet,  
    eval_dataset=test_dataset_triplet,    
    data_collator=triplet_collate_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3), MemoryCleanupCallback()],
)

# ------------------------------------------------------------------------------
# 4. CHẠY VÒNG LẶP HUẤN LUYỆN & VẼ CHART LOSS
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
    plt.savefig(os.path.join(output_dir, "training_charts.png"))
    plt.close()

def plot_cbir_metrics(acc, prec, rec, map_score, output_dir):
    print(f"[REPORT] Generating CBIR Final Metrics Chart...")
    metrics = ['Accuracy@5', 'Precision@5', 'Recall@5', 'mAP@5']
    values = [acc, prec, rec, map_score]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, values, color=colors)
    plt.title('Final CBIR Performance on Cifar-100 Test Set')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 110)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cbir_metrics_chart.png"))
    plt.close()

print("\n" + "="*60)
print("--> STARTING CIFAR-100 RESNET TRIPLET LOSS TRAINING LOOP")
print("="*60 + "\n")

torch.cuda.empty_cache()
trainer.train()
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
plot_training_history(trainer.state.log_history, OUTPUT_DIR)

# ------------------------------------------------------------------------------
# 5. ĐÁNH GIÁ CHỈ SỐ CBIR VÀ VẼ BIỂU ĐỒ BÁO CÁO CỘT
# ------------------------------------------------------------------------------
print("\n" + "="*60)
print("--> BẮT ĐẦU ĐÁNH GIÁ ĐỘ ĐO CBIR TẬP TEST (10.000 ẢNH)")
print("="*60)
eval_dataset = PureCifarTripletDataset(test_imgs_np, test_labels, _val_tf, is_train=False)
from torch.utils.data import DataLoader
# Tăng Batch Size lên cực hạn (128) để bứt tốc quá trình trích xuất Vector từ 10.000 ảnh Test (Vì không cần Backward Pass nên VRAM rất rảnh)
test_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)

model.to(DEVICE)
model.eval()
feature_vectors = []
labels_list = []

print("Đang trích xuất vector từ tập Test...")
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Extracting Features"):
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels_batch = batch["labels"].numpy()
        
        outputs = model(pixel_values)
        embeddings = outputs.pooler_output.flatten(1) # RÚT TỪ POOLER OUTPUT CHUẨN CỦA RESNET
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        feature_vectors.extend(embeddings.cpu().numpy())
        labels_list.extend(labels_batch)

feature_vectors = np.array(feature_vectors)
labels_array = np.array(labels_list)

print("Đang tính toán ma trận Cosine và các độ đo (Matrix Size: 10000x10000)...")
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
print("🚀 BÁO CÁO KẾT QUẢ CBIR RESNET TRÊN CIFAR-100")
print("="*60)
print(f"Accuracy@{TOP_K}:  {acc_final:.2f}%")
print(f"Precision@{TOP_K}: {prec_final:.2f}%")
print(f"Recall@{TOP_K}:    {rec_final:.2f}%")
print(f"mAP@{TOP_K}:       {map_final:.2f}%")
print("="*60)

plot_cbir_metrics(acc_final, prec_final, rec_final, map_final, OUTPUT_DIR)
print(f"✅ Biểu đồ đánh giá CBIR đã được lưu tại: {OUTPUT_DIR}/cbir_metrics_chart.png")
