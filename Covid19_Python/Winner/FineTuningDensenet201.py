import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. KLASÖR YOLLARI VE AYARLAR
# ============================================================
dataset_raw_dir = r"C:\Users\ThinkPad\Desktop\Makine\Covid19-3\Dataset"
output_dir = r"C:\Users\ThinkPad\Desktop\Makine\Covid19-3\Densenet201_feature_Selection"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

batch_size = 32
epochs_warmup = 3       
epochs_finetune = 7     
num_classes = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. EĞİTİM MOTORU (FONKSİYON OLARAK DIŞARIDA TANIMLANMALI)
# ============================================================
def train_model(model, dataloaders, criterion, optimizer, num_epochs, phase_name):
    for epoch in range(num_epochs):
        print(f"   -> {phase_name} Tur {epoch+1}/{num_epochs} | ", end="")
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase.upper()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} ", end=" | ")
        print()
    return model

# ============================================================
# 3. ANA ÇALIŞMA BLOĞU (WINDOWS MULTIPROCESSING KORUMASI)
# ============================================================
if __name__ == '__main__':
    print("==================================================================")
    print("🔥 DENSENET201 PYTORCH FINE-TUNING & DATA AUGMENTATION 🔥")
    print(f"-> Kullanılan Donanım: {device}")
    print("==================================================================")

    print("\n[1/5] Tıbbi görüntülere uygun dönüşümler (Transforms) ayarlanıyor...")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    full_dataset = datasets.ImageFolder(dataset_raw_dir)

    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']

    # Burada num_workers=2 kullanıyoruz, if __name__ == '__main__' sayesinde artık çökmeyecek!
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    }

    print("\n[2/5] PyTorch DenseNet201 tabanı yükleniyor...")
    model = models.densenet201(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier.in_features  
    model.classifier = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # ----------------- WARM-UP AŞAMASI -----------------
    print(f"\n[3/5] WARM-UP AŞAMASI: Sadece yeni başlık eğitiliyor ({epochs_warmup} Tur)...")
    optimizer_warmup = optim.Adam(model.classifier.parameters(), lr=1e-3)
    model = train_model(model, dataloaders, criterion, optimizer_warmup, epochs_warmup, "Isınma")

    # ----------------- FINE-TUNING AŞAMASI -----------------
    print(f"\n[4/5] FINE-TUNING AŞAMASI: Son Dense Block buzları çözülüyor ({epochs_finetune} Tur)...")
    for name, param in model.named_parameters():
        if "features.denseblock4" in name or "features.norm5" in name:
            param.requires_grad = True

    optimizer_finetune = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    model = train_model(model, dataloaders, criterion, optimizer_finetune, epochs_finetune, "Uzmanlaşma")

    # ============================================================
    # 4. YENİ KESKİN ÖZELLİKLERİN ÇIKARILMASI
    # ============================================================
    print("\n[5/5] Akciğer uzmanı PyTorch modeli üzerinden rafine özellikler çıkarılıyor...")

    extract_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model.eval()
    all_features = []
    all_labels = []

    start_extract = time.time()
    with torch.no_grad():
        for inputs, labels in extract_loader:
            inputs = inputs.to(device)
            
            features = model.features(inputs)
            out = nn.functional.relu(features, inplace=True)
            out = nn.functional.adaptive_avg_pool2d(out, (1, 1)) 
            out = torch.flatten(out, 1) 
            
            all_features.append(out.cpu().numpy())
            all_labels.append(labels.numpy())

    X_features = np.vstack(all_features)
    y_labels = np.concatenate(all_labels)

    print(f"-> Özellik çıkarma süresi: {time.time() - start_extract:.2f} saniye")

    # ============================================================
    # 5. .NPY DOSYALARINI DİSKE YAZMA
    # ============================================================
    features_save_path = os.path.join(output_dir, 'X_features_densenet201_pytorch_finetuned.npy')
    labels_save_path = os.path.join(output_dir, 'y_labels_pytorch_finetuned.npy')

    np.save(features_save_path, X_features)
    np.save(labels_save_path, y_labels)

    print("\n==================================================================")
    print("🎯 PYTORCH OPERASYONU BAŞARIYLA TAMAMLANDI!")
    print(f"-> Yeni Özellik Matrisi : {X_features.shape} -> Şuraya kaydedildi:\n   {features_save_path}")
    print(f"-> Yeni Etiket Matrisi  : {y_labels.shape} -> Şuraya kaydedildi:\n   {labels_save_path}")
    print("==================================================================")