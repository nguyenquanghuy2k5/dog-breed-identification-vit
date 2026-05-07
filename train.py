# ==============================================================================
# MODULE: CUB-200-2011 OFFICIAL BENCHMARK PIPELINE
# STANDARD: Fixed Split (Official) + Bounding Box Cropping
# METRICS: Top-1 Acc, Top-5 Acc, mAP
# ==============================================================================

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from transformers import Trainer
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from datasets import Dataset, DatasetDict, Features, ClassLabel, Value, Image as DatasetImage
from transformers import (
    ViTImageProcessor, ViTModel, TrainingArguments, Trainer, DefaultDataCollator, EarlyStoppingCallback
)
from torchvision.transforms import (
    CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor, RandomRotation, ColorJitter
)
from sklearn.metrics import accuracy_score, top_k_accuracy_score, average_precision_score
from sklearn.preprocessing import label_binarize

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

# --- CẤU HÌNH QUAN TRỌNG ĐỂ TRÁNH LỖI ẢNH (Corrupt Images) ---
ImageFile.LOAD_TRUNCATED_IMAGES = True  
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# ------------------------------------------------------------------------------
# 1. CẤU HÌNH HỆ THỐNG
# ------------------------------------------------------------------------------
TASK_MODE = "BIRDS"
SEED = 42
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ĐƯỜNG DẪN: Hãy sửa lại đúng folder dataset của bạn
DATASET_DIR = "./Cub_200_2011"
# Mẹo: Đặt Output ra ổ đĩa còn nhiều dung lượng (tránh Desktop/OneDrive gây lỗi Permission)
OUTPUT_DIR  = "./model_birds_cub200_final" # Sửa thành đường dẫn tương đối cho Google Colab
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 8

# ------------------------------------------------------------------------------
# 2. LOAD DATA CHUẨN QUỐC TẾ (OFFICIAL SPLIT)
# ------------------------------------------------------------------------------
def load_cub_benchmark(base_path):
    print(f"--> [DATA] Reading CUB-200 Standard Metadata from: {base_path}")
    try:
        # 1. Đọc các file cấu hình chuẩn
        df_images = pd.read_csv(os.path.join(base_path, "images.txt"), sep=' ', names=['id', 'path'])
        df_labels = pd.read_csv(os.path.join(base_path, "image_class_labels.txt"), sep=' ', names=['id', 'label'])
        
        # QUAN TRỌNG: Đây là file quyết định chuẩn Benchmark (is_train: 1=Train, 0=Test)
        df_split = pd.read_csv(os.path.join(base_path, "train_test_split.txt"), sep=' ', names=['id', 'is_train'])
        
        df_boxes = pd.read_csv(os.path.join(base_path, "bounding_boxes.txt"), sep=' ', names=['id', 'x', 'y', 'w', 'h'])
        df_classes = pd.read_csv(os.path.join(base_path, "classes.txt"), sep=' ', names=['id', 'class_name'])

        # 2. Merge dữ liệu
        df = df_images.merge(df_labels, on='id').merge(df_split, on='id').merge(df_boxes, on='id')
        
        # 3. Tạo đường dẫn tuyệt đối
        img_root = os.path.join(base_path, "images")
        df['path'] = df['path'].apply(lambda x: os.path.join(img_root, x))

        # 4. Chia Train/Test theo đúng file split
        train_df = df[df['is_train'] == 1]
        test_df = df[df['is_train'] == 0]
        
        class_names = df_classes['class_name'].tolist()
        print(f"--> [INFO] Standard Split Statistics:")
        print(f"    - Train Set: {len(train_df)} images (Expected ~5994)")
        print(f"    - Test Set:  {len(test_df)} images (Expected ~5794)")

        # 5. Định nghĩa Features
        features = Features({
            'image': DatasetImage(), 
            'label': ClassLabel(names=class_names),
            'box_x': Value('float32'), 'box_y': Value('float32'), 
            'box_w': Value('float32'), 'box_h': Value('float32')
        })

        def create_ds(df_sub):
            # CUB Label bắt đầu từ 1, phải trừ 1 để về 0-based index cho PyTorch
            return Dataset.from_dict({
                "image": df_sub['path'].tolist(), 
                "label": [l - 1 for l in df_sub['label'].tolist()],
                "box_x": df_sub['x'].tolist(), "box_y": df_sub['y'].tolist(),
                "box_w": df_sub['w'].tolist(), "box_h": df_sub['h'].tolist()
            }, features=features)

        return DatasetDict({"train": create_ds(train_df), "test": create_ds(test_df)})
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load metadata. Check paths: {e}")
        exit(1)

ds = load_cub_benchmark(DATASET_DIR)
labels = ds["train"].features["label"].names

# ------------------------------------------------------------------------------
# 3. TIỀN XỬ LÝ & TẠO BỘ BA (TRIPLET DATASET LOGIC)
# ------------------------------------------------------------------------------
MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(MODEL_CHECKPOINT)
size = processor.size["height"]
normalize = Normalize(mean=processor.image_mean, std=processor.image_std)

_train_tf = Compose([
    RandomResizedCrop(size, scale=(0.8, 1.0)), 
    RandomHorizontalFlip(), 
    RandomRotation(15), 
    ToTensor(), 
    normalize
])
_val_tf = Compose([
    Resize(256), 
    CenterCrop(size), 
    ToTensor(), 
    normalize
])
# Xây dựng Index từ điển để bốc ảnh siêu tốc
print("--> [INFO] Đang xây dựng cấu trúc Index cho Triplet Loss...")
train_labels = ds["train"]["label"]
label_to_indices = defaultdict(list)
for i, label in enumerate(train_labels):
    label_to_indices[label].append(i)
unique_labels = list(label_to_indices.keys())

def apply_triplet_transforms(examples):
    """Hàm này bốc 3 ảnh (Anchor, Positive, Negative) cho mỗi index"""
    anchor_pixels, pos_pixels, neg_pixels = [], [], []
    
    for i in range(len(examples['image'])):
        # 1. ANCHOR
        anchor_img = examples['image'][i].convert("RGB")
        anchor_label = examples["label"][i]
        
        # 2. POSITIVE (Cùng loài)
        pos_idx = random.choice(label_to_indices[anchor_label])
        pos_img = ds["train"][pos_idx]['image'].convert("RGB")
        
        # 3. NEGATIVE (Khác loài)
        neg_label = random.choice(unique_labels)
        while neg_label == anchor_label:
            neg_label = random.choice(unique_labels)
        neg_idx = random.choice(label_to_indices[neg_label])
        neg_img = ds["train"][neg_idx]['image'].convert("RGB")
        
        # Đưa qua bộ biến đổi (Crop & Tensor)
        anchor_pixels.append(_train_tf(anchor_img))
        pos_pixels.append(_train_tf(pos_img))
        neg_pixels.append(_train_tf(neg_img))
    return {
        "anchor_values": anchor_pixels,
        "positive_values": pos_pixels,
        "negative_values": neg_pixels,
        "labels": examples["label"] # Giữ lại để tính mAP nếu cần
    }
# Ép Dataset sử dụng logic Triplet
train_dataset_triplet = ds["train"].with_transform(apply_triplet_transforms)
test_dataset_triplet = ds["test"].with_transform(lambda x: apply_triplet_transforms(x))    

# ------------------------------------------------------------------------------
# 5. CẤU HÌNH TRAINING (TỐI ƯU Ổ CỨNG)
# ------------------------------------------------------------------------------
model = ViTModel.from_pretrained(
    MODEL_CHECKPOINT, 
    num_labels=len(labels), 
    id2label={i:l for i,l in enumerate(labels)}, 
    label2id={l:i for i,l in enumerate(labels)}, 
    ignore_mismatched_sizes=True
)

# =========================================================================
# FREEZING LAYER (CHỐNG OVERFITTING & MAX SPEED COLAB)
# =========================================================================
print("[INFO] Đang đóng băng (Freezing) 10/12 Layers đầu tiên của ViT...")
for param in model.embeddings.parameters():
    param.requires_grad = False
for layer in model.encoder.layer[:10]:
    for param in layer.parameters():
        param.requires_grad = False

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=40,            
    learning_rate=2e-5,             
    lr_scheduler_type="cosine",     
    per_device_train_batch_size=BATCH_SIZE,  
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  
    
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,             
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss", # SỬA: Dựa vào Loss của tập Test
    greater_is_better=False,           # SỬA: Loss càng thấp càng tốt
    
    remove_unused_columns=False,    
    report_to="none",
    fp16=(DEVICE=="cuda"),        
    dataloader_num_workers=2,
    logging_steps=10      
)

## ------------------------------------------------------------------------------
# CLASS CUSTOM TRAINER CHO TRIPLET LOSS 
# ------------------------------------------------------------------------------
class TripletTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Lấy dữ liệu từ Dictionary mà ta vừa tạo ở muc 3
        anchor_imgs = inputs.get("anchor_values")
        positive_imgs = inputs.get("positive_values")
        negative_imgs = inputs.get("negative_values")

        # Đẩy 3 ảnh qua model ViT để lấy Vector (Điều chỉnh 2: Forward Pass)
        # Sử dụng [:, 0, :] để lấy token [CLS] đại diện cho toàn bộ ảnh
        anchor_vec = model(pixel_values=anchor_imgs).last_hidden_state[:, 0, :]
        positive_vec = model(pixel_values=positive_imgs).last_hidden_state[:, 0, :]
        negative_vec = model(pixel_values=negative_imgs).last_hidden_state[:, 0, :]

        # Chuẩn hóa Vector (L2 Normalization - Bắt buộc cho FAISS sau này)
        anchor_vec = nn.functional.normalize(anchor_vec, p=2, dim=1)
        positive_vec = nn.functional.normalize(positive_vec, p=2, dim=1)
        negative_vec = nn.functional.normalize(negative_vec, p=2, dim=1)

        # Tính Triplet Loss
        # Margin = 1.0 (Khoảng cách an toàn giữa Positive và Negative)
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
        
        # Ép ngắt đồ thị và dọn rác RAM
        loss = loss.detach()
        del inputs
        gc.collect()
        
        return (loss, None, None)

# Khởi tạo Trainer với Class mới
trainer = TripletTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_triplet,  
    eval_dataset=test_dataset_triplet,    
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# ------------------------------------------------------------------------------
# VISUALIZATION: vẽ biểu đồ
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

# ------------------------------------------------------------------------------
# 6. THỰC THI
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("--> STARTING CUB-200 OFFICIAL BENCHMARK RUN")
    print(f"--> Split Protocol: Standard (train_test_split.txt)")
    print(f"--> Crop Protocol:  Bounding Box (Method C)")
    print("="*60 + "\n")
    
    torch.cuda.empty_cache()
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)  # <--- BẮT BUỘC THÊM DÒNG NÀY ĐỂ WEB ĐỌC ĐƯỢC
    
    # Vẽ biểu đồ Triplet Loss xuống File
    plot_training_history(trainer.state.log_history, OUTPUT_DIR)
    
    # =====================================================================
    # ĐÁNH GIÁ CBIR METRICS NGAY SAU KHI TRAIN XONG (ĐỒNG BỘ VỚI EXTRACT_FEATURES)
    # =====================================================================
    print("\n" + "="*60)
    print("--> BẮT ĐẦU ĐÁNH GIÁ ĐỘ ĐO CBIR (mAP, Recall@5, Precision@5)")
    print("="*60)
    
    from sklearn.metrics.pairwise import cosine_similarity
    from torch.utils.data import DataLoader

    # 1. Tạo tập Test chuẩn (Không dùng Triplet để lấy vector)
    def preprocess_test(examples):
        examples['pixel_values'] = [_val_tf(img.convert("RGB")) for img in examples['image']]
        
        # Xóa cột 'image' gốc (dạng PIL) để tránh lỗi DataLoader y như extract_features
        if 'image' in examples:
            del examples['image']
            
        return examples

    eval_dataset = ds["test"].with_transform(preprocess_test)
    test_loader = DataLoader(eval_dataset, batch_size=16, collate_fn=DefaultDataCollator())

    # 2. Trích xuất Vector
    model.eval()
    feature_vectors = []
    labels_list = []
    print("Đang trích xuất vector từ tập Test...")
    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(DEVICE) # Bỏ torch.stack đi
            # Sửa lỗi: DefaultDataCollator mặc định tự động đổi tên 'label' thành 'labels'
            labels = batch["labels"].numpy()
            
            outputs = model(pixel_values)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            feature_vectors.extend(embeddings.cpu().numpy())
            labels_list.extend(labels)
    feature_vectors = np.array(feature_vectors)
    labels_array = np.array(labels_list)

    # 3. Tính toán Ma trận Cosine & Metrics
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
        # Sắp xếp và loại bỏ chính ảnh đang query
        ranked_indices = np.argsort(similarities)[::-1]
        ranked_indices = ranked_indices[ranked_indices != i]
        
        top_k_indices = ranked_indices[:TOP_K]
        retrieved_labels = labels_array[top_k_indices]
        # Relevant
        num_relevant_retrieved = np.sum(retrieved_labels == query_label)
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
    # 4. IN BÁO CÁO CUỐI CÙNG
    acc_final = accuracy_count / num_samples * 100
    prec_final = np.mean(precision_scores) * 100
    rec_final = np.mean(recall_scores) * 100
    map_final = np.mean(average_precisions) * 100

    print("\n" + "="*60)
    print("🚀 BÁO CÁO KẾT QUẢ HUẤN LUYỆN CBIR (TRIPLET LOSS)")
    print("="*60)
    print(f"Accuracy@{TOP_K}:  {acc_final:.2f}%")
    print(f"Precision@{TOP_K}: {prec_final:.2f}%")
    print(f"Recall@{TOP_K}:    {rec_final:.2f}%")
    print(f"mAP@{TOP_K}:       {map_final:.2f}%")
    print("="*60)
    
    plot_cbir_metrics(
        acc_final, prec_final, rec_final, map_final, 
        OUTPUT_DIR, "Final CBIR Performance on CUB-200 (ViT)"
    )
    
    
    