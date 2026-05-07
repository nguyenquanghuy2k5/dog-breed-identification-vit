import os
import torch
import faiss
import numpy as np
import pandas as pd
from PIL import Image
from transformers import ResNetModel, AutoImageProcessor
import torchvision.transforms as transforms

BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_dogs_resnet50_baseline")
PATHS_FILE_CSV = os.path.join(BASE_DIR, "models", "databases", "model_dogs_resnet50_baseline_info.csv")
PATHS_FILE_NPY = os.path.join(BASE_DIR, "models", "databases", "model_dogs_resnet50_baseline_info.npy")
INDEX_FILE = os.path.join(BASE_DIR, "models", "databases", "model_dogs_resnet50_baseline_faiss.index")

# 1. FIX TẬN GỐC FILE JSON
print("1. Đang tự động tạo lại preprocessor_config.json chuẩn quốc tế...")
try:
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    processor.save_pretrained(MODEL_PATH)
    print("-> ✅ Đã vá xong lỗi JSON!")
except Exception as e:
    print(f"Lỗi tải JSON: {e}")

# 2. ĐỌC DANH SÁCH ẢNH
print("\n2. Đang nạp danh sách đường dẫn ảnh từ file Info...")
if os.path.exists(PATHS_FILE_CSV):
    df = pd.read_csv(PATHS_FILE_CSV)
    image_paths = df['filepath'].values
else:
    image_paths = np.load(PATHS_FILE_NPY, allow_pickle=True)
print(f"-> Đã tìm thấy {len(image_paths)} bức ảnh chó.")

# 3. KHỞI TẠO MÔ HÌNH VÀ FAISS
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n3. Đang tải mô hình ResNet-50 lên {device}...")
model = ResNetModel.from_pretrained(MODEL_PATH).to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dimension = 2048
index = faiss.IndexFlatIP(dimension)

# 4. CHẠY LẠI TOÀN BỘ DỮ LIỆU
print("\n4. BẮT ĐẦU ÉP KIỂU LẠI TOÀN BỘ CƠ SỞ DỮ LIỆU (Có thể mất vài phút)...")
vectors = []
with torch.no_grad():
    for i, path in enumerate(image_paths):
        if i % 1000 == 0 and i > 0:
            print(f"   ⏳ Đã xử lý {i}/{len(image_paths)} ảnh...")
        try:
            image = Image.open(path).convert("RGB")
            inputs = transform(image).unsqueeze(0).to(device)
            outputs = model(pixel_values=inputs)
            feat = outputs.pooler_output.flatten(1)
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            vectors.append(feat.cpu().numpy().astype('float32')[0])
        except Exception as e:
            print(f"   [Cảnh báo] Lỗi tại ảnh {path}: {e}")
            vectors.append(np.zeros(dimension, dtype='float32'))

print("\n5. Đang lưu CSDL mới vào đĩa cứng...")
vector_matrix = np.vstack(vectors)
index.add(vector_matrix)
faiss.write_index(index, INDEX_FILE)

print("\n🎉 HOÀN TẤT! CƠ SỞ DỮ LIỆU ĐÃ ĐƯỢC TẨY SẠCH VÀ LÀM MỚI!")
print("Bây giờ bạn hãy ra trình duyệt và F5 tải lại trang Web nhé!")