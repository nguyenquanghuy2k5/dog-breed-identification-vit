# ==============================================================================
# MODULE: STANFORD DOGS BENCHMARK TRAINING
# MODE: TRIPLET LOSS + BOUNDING BOX CROP + CBIR REPORT GENERATION
# ==============================================================================

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
    ViTImageProcessor, ViTModel, TrainingArguments, Trainer, DefaultDataCollator, EarlyStoppingCallback
)
from torchvision.transforms import (
    CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor, RandomRotation, ColorJitter
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

# ------------------------------------------------------------------------------
# SYSTEM CONFIGURATION
# ------------------------------------------------------------------------------
# Using local paths suitable for Google Colab/local system
DATA_ROOT = "/kaggle/working/Stanford_Dogs_Data"
OUTPUT_DIR = "/kaggle/working/model_dogs_stanford_final"

URL_IMAGES = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
URL_ANNOTATIONS = "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"
URL_LISTS = "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"

MODEL_NAME = "google/vit-base-patch16-224-in21k"
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 8
EPOCHS = 20
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Allow loading truncated images to prevent errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ------------------------------------------------------------------------------
# DATASET IMPLEMENTATION
# ------------------------------------------------------------------------------
class StanfordDogsDataset(Dataset):
    def __init__(self, root, split="train", processor=None, mode="triplet"):
        """
        mode: "triplet" yields (anchor, positive, negative).
              "eval" yields standard single images for validation.
        """
        self.root = root
        self.split = split
        self.processor = processor
        self.mode = mode
        self.images_dir = os.path.join(root, "Images")
        self.annots_dir = os.path.join(root, "Annotation")
        
        self._prepare_data()
        
        print(f"[INFO] Loading {split} metadata from .mat files...")
        mat_file = "train_list.mat" if split == "train" else "test_list.mat"
        mat = scipy.io.loadmat(os.path.join(root, mat_file))
        
        self.file_list = [f[0][0] for f in mat['file_list']]
        self.labels = [l[0] - 1 for l in mat['labels']]
        unique_classes = sorted(list(set([f.split('/')[0].split('-')[1] for f in self.file_list])))
        self.classes = unique_classes
        
        size = processor.size["height"]
        normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
        
        if split == "train":
            self.transform = Compose([
                RandomResizedCrop(size, scale=(0.8, 1.0)),
                RandomHorizontalFlip(),
                RandomRotation(15),
                ColorJitter(brightness=0.2, contrast=0.2),
                ToTensor(),
                normalize,
            ])
        else:
            self.transform = Compose([
                Resize(256),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ])

        # Prepare Triplet Indices for "triplet" mode lookup
        if self.mode == "triplet":
            print(f"--> [INFO] Indexing Triplet Pairs for {split}...")
            self.label_to_indices = defaultdict(list)
            for i, label in enumerate(self.labels):
                self.label_to_indices[label].append(i)
            self.unique_labels = list(self.label_to_indices.keys())

    def _prepare_data(self):
        os.makedirs(self.root, exist_ok=True)
        
        def download_extract(url, filename):
            f_path = os.path.join(self.root, filename)
            if not os.path.exists(f_path):
                print(f"[DOWNLOAD] Starting download: {filename}")
                with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                    urlretrieve(url, f_path, reporthook=lambda b, bsize, tsize: t.update(b*bsize - t.n))
                
                print(f"[EXTRACT] Extracting archive: {filename}")
                with tarfile.open(f_path, "r") as tar:
                    tar.extractall(path=self.root)
            else:
                pass 

        download_extract(URL_IMAGES, "images.tar")
        download_extract(URL_ANNOTATIONS, "annotation.tar")
        download_extract(URL_LISTS, "lists.tar")

    def _get_bbox(self, path_wo_ext):
        xml_path = os.path.join(self.annots_dir, path_wo_ext)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            b = root.find('object').find('bndbox')
            return (
                int(b.find('xmin').text),
                int(b.find('ymin').text),
                int(b.find('xmax').text),
                int(b.find('ymax').text)
            )
        except:
            return None

    def _load_image(self, idx):
        file_path = self.file_list[idx]
        image_path = os.path.join(self.images_dir, file_path)
        image = Image.open(image_path).convert("RGB")
        bbox = self._get_bbox(file_path.rsplit('.', 1)[0])
        if bbox:
            image = image.crop(bbox)
        return image

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        anchor_label = self.labels[idx]
        
        if self.mode == "triplet":
            # 1. ANCHOR
            anchor_img = self._load_image(idx)
            
            # 2. POSITIVE (Cùng loài)
            pos_idx = random.choice(self.label_to_indices[anchor_label])
            pos_img = self._load_image(pos_idx)
            
            # 3. NEGATIVE (Khác loài)
            neg_label = random.choice(self.unique_labels)
            while neg_label == anchor_label:
                neg_label = random.choice(self.unique_labels)
            neg_idx = random.choice(self.label_to_indices[neg_label])
            neg_img = self._load_image(neg_idx)
            
            return {
                "anchor_values": self.transform(anchor_img),
                "positive_values": self.transform(pos_img),
                "negative_values": self.transform(neg_img),
                "labels": torch.tensor(anchor_label, dtype=torch.long)
            }
        else:
            # Eval mode (standard features extraction)
            img = self._load_image(idx)
            return {
                "pixel_values": self.transform(img),
                "labels": torch.tensor(anchor_label, dtype=torch.long)
            }

# ------------------------------------------------------------------------------
# CLASS CUSTOM TRAINER CHO TRIPLET LOSS 
# ------------------------------------------------------------------------------
class TripletTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        anchor_imgs = inputs.get("anchor_values")
        positive_imgs = inputs.get("positive_values")
        negative_imgs = inputs.get("negative_values")

        # Forward Pass qua model ViT để lấy Vector (Lấy token [CLS] đại diện)
        anchor_vec = model(pixel_values=anchor_imgs).last_hidden_state[:, 0, :]
        positive_vec = model(pixel_values=positive_imgs).last_hidden_state[:, 0, :]
        negative_vec = model(pixel_values=negative_imgs).last_hidden_state[:, 0, :]

        # L2 Normalization (Bắt buộc cho khoảng cách mảng Triplet/FAISS)
        anchor_vec = nn.functional.normalize(anchor_vec, p=2, dim=1)
        positive_vec = nn.functional.normalize(positive_vec, p=2, dim=1)
        negative_vec = nn.functional.normalize(negative_vec, p=2, dim=1)

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
        
        # Ép ngắt đồ thị và dọn rác RAM
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

# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------
def main():
    print(f"[INIT] System initialized. Device: {DEVICE.upper()}")
    
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    
    # Init Datasets for Training (Triplet) and Eval (Triplet for Trainer validation step)
    train_dataset = StanfordDogsDataset(DATA_ROOT, split="train", processor=processor, mode="triplet")
    test_dataset_triplet = StanfordDogsDataset(DATA_ROOT, split="test", processor=processor, mode="triplet")
    
    id2label = {str(i): c for i, c in enumerate(train_dataset.classes)}
    label2id = {c: str(i) for i, c in enumerate(train_dataset.classes)}

    print(f"[DATA] Train Set: {len(train_dataset)} | Test Set: {len(test_dataset_triplet)}")
    print(f"[DATA] Classes: {len(train_dataset.classes)}")

    model = ViTModel.from_pretrained(
        MODEL_NAME,
        num_labels=len(train_dataset.classes),
        id2label=id2label,
        label2id=label2id,
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

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=(DEVICE=="cuda"),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        dataloader_num_workers=2 # Optimized for Colab 2 CPU Cores
    )

    trainer = TripletTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset_triplet,
        data_collator=DefaultDataCollator(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("\n" + "="*60)
    print("--> STARTING STANFORD DOGS TRIPLET LOS TRAINING LOOP")
    print("="*60 + "\n")
    
    torch.cuda.empty_cache()
    trainer.train()
    
    print("[PROCESS] Saving final model artifacts...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    plot_training_history(trainer.state.log_history, OUTPUT_DIR)
    
    # =====================================================================
    # ĐÁNH GIÁ CBIR METRICS NGAY SAU KHI TRAIN XONG 
    # =====================================================================
    print("\n" + "="*60)
    print("--> BẮT ĐẦU ĐÁNH GIÁ ĐỘ ĐO CBIR (mAP, Recall@5, Precision@5)")
    print("="*60)
    
    # 1. Khởi tạo Dataset chế độ đánh giá thường (trả về 1 pixel_values)
    eval_dataset = StanfordDogsDataset(DATA_ROOT, split="test", processor=processor, mode="eval")
    test_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

    # 2. Trích xuất Vector
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
        
        # Relevant items mask
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

    # 4. IN BÁO CÁO CUỐI CÙNG
    acc_final = accuracy_count / num_samples * 100
    prec_final = np.mean(precision_scores) * 100
    rec_final = np.mean(recall_scores) * 100
    map_final = np.mean(average_precisions) * 100

    print("\n" + "="*60)
    print("🚀 BÁO CÁO KẾT QUẢ HUẤN LUYỆN CBIR STANFORD DOGS")
    print("="*60)
    print(f"Accuracy@{TOP_K}:  {acc_final:.2f}%")
    print(f"Precision@{TOP_K}: {prec_final:.2f}%")
    print(f"Recall@{TOP_K}:    {rec_final:.2f}%")
    print(f"mAP@{TOP_K}:       {map_final:.2f}%")
    print("="*60)

    plot_cbir_metrics(
        acc_final, prec_final, rec_final, map_final, 
        OUTPUT_DIR, "Final CBIR Performance on Stanford Dogs (ViT)"
    )

if __name__ == "__main__":
    main()