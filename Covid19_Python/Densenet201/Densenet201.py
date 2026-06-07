import os
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from glob import glob
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. KLASÖR YOLLARI
# ============================================================
dataset_dir = r"C:\Users\ThinkPad\Desktop\Makine\Covid19-3\dataset"
save_dir = r"C:\Users\ThinkPad\Desktop\Makine\Covid19-3\Densenet201_feature_Selection"

os.makedirs(save_dir, exist_ok=True)

# ============================================================
# 2. DONANIM KONTROLÜ
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"==================================================")
print(f"🚀 KULLANILAN MOTOR: {device}")
print(f"==================================================\n")

# ============================================================
# 3. KUSURSUZ VERİ YÜKLEYİCİ (TÜM VERİ TEK PARÇA)
# ============================================================
class CovidRawDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
        self.image_paths = []
        self.labels = []
        
        for label_idx, cls_name in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls_name, '**', '*.png')
            files = glob(cls_path, recursive=True)
            
            if len(files) == 0:
                cls_path = os.path.join(root_dir, cls_name, '*.png')
                files = glob(cls_path)
                
            for f in files:
                self.image_paths.append(f)
                self.labels.append(label_idx)
                
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ŞAMPİYONLUK DOKUNUŞU: Orijinal 299x299 boyutu korunuyor!
transform = transforms.Compose([
    transforms.Resize((299, 299)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Tüm veri seti taranıyor...")
full_dataset = CovidRawDataset(root_dir=dataset_dir, transform=transform)
print(f"Sınıflar: {full_dataset.classes}")
print(f"Toplam Bulunan Resim: {len(full_dataset)}")

# GPU hafıza taşmasını önlemek için batch_size=32 yapıldı
batch_size = 32 
full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ============================================================
# 4. DENSENET-201 MİMARİSİ (Özellik Çıkarıcı Modifikasyonu)
# ============================================================
print("\nDenseNet-201 Yükleniyor...")

class DenseNet201Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        densenet = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        self.features = densenet.features
        
    def forward(self, x):
        x = self.features(x)
        x = nn.functional.relu(x, inplace=True)
        # Resim boyutu ne olursa olsun (299x299), AdaptivePool onu standart 1x1'e sıkıştırıp 1920 vektörü verir.
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x

base_model = DenseNet201Extractor().to(device)
base_model.eval()

# ============================================================
# 5. BÜYÜK ÖZELLİK ÇIKARIM DÖNGÜSÜ
# ============================================================
def extract_all_features(loader):
    print(f"\nTüm Veri Seti İçin Yüksek Çözünürlüklü (299x299) Özellik Çıkarımı Başlıyor...")
    features_list = []
    labels_list = []
    start_time = time.time()
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            outputs = base_model(inputs) 
            
            features_list.append(outputs.cpu().numpy())
            labels_list.append(targets.numpy())
            
            # Daha sık bilgi versin diye 20 batch'te bir yazdır
            if (i + 1) % 20 == 0:
                print(f"  İşlenen Batch: {i + 1}/{len(loader)}")
                
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    print(f"\nTamamlandı! Toplam Süre: {(time.time() - start_time)/60:.2f} dk | Çıkan Matris: {features.shape}")
    return features, labels

X_all, y_all = extract_all_features(full_loader)

# ============================================================
# 6. KAYIT İŞLEMİ (TEK PARÇA NPY FORMATINDA)
# ============================================================
print(f"\nÇıkarılan devasa matris {save_dir} dizinine kaydediliyor...")

np.save(os.path.join(save_dir, 'X_features_densenet201.npy'), X_all)
np.save(os.path.join(save_dir, 'y_labels.npy'), y_all)
np.save(os.path.join(save_dir, 'class_names.npy'), np.array(full_dataset.classes))

print("==================================================")
print("✔ İŞLEM TAMAM! 299x299 Orijinal kalite özellikleri tek bir matriste toplandı.")
print("==================================================")