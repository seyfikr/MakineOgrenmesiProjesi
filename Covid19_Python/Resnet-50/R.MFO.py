import os
import time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import math

warnings.filterwarnings("ignore")

# ============================================================
# 1. KLASÖR YOLLARI VE VERİ YÜKLEME (RESNET-50 KARARGAHI)
# ============================================================
work_dir = r"C:\Users\ThinkPad\Desktop\Makine\Covid19-3\ResNet50_feature_Selection"

print("==================================================")
print("🦋 MFO (PERVANE-ATEŞ) RESNET-50 ÖZELLİK SEÇİMİ 🦋")
print("==================================================")

print("\nResNet-50 verileri diskten okunuyor...")
X_all = np.load(os.path.join(work_dir, 'X_features_resnet50.npy'))
y_all = np.load(os.path.join(work_dir, 'y_labels.npy'))

print(f"Toplam Veri: {X_all.shape[0]} Resim | Özellik Sayısı (Uzay): {X_all.shape[1]}")

# Adil Veri Bölme (Orijinal Uyum)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=42, stratify=y_all)

# ============================================================
# 2. MFO PARAMETRELERİ (10 Pervane, 10 Tur, 4 Erken Durdurma)
# ============================================================
num_moths = 10
max_iter = 10
patience_limit = 4  
num_features = X_train.shape[1]

# Pervanelerin rastgele başlatılması
Moth_Positions = np.random.randint(2, size=(num_moths, num_features)).astype(float)
Moth_Scores = np.zeros(num_moths)

# Ateşler (Flames)
Flame_Positions = np.zeros((num_moths, num_features))
Flame_Scores = np.zeros(num_moths)

cache = {}

# ============================================================
# 3. FİTNESS (Uygunluk) FONKSİYONU
# ============================================================
def fitness(position):
    bin_pos = np.where(position >= 0.5, 1, 0)
    key = tuple(bin_pos)
    if key in cache: 
        return cache[key]
    
    selected_features = np.where(bin_pos == 1)[0]
    
    if len(selected_features) == 0: 
        return 0.0
    
    # Hızlı Karar Verici XGBoost
    clf = xgb.XGBClassifier(n_estimators=20, max_depth=3, learning_rate=0.1, 
                            tree_method='hist', random_state=42, n_jobs=-1)
    
    clf.fit(X_train[:, selected_features], y_train)
    preds = clf.predict(X_test[:, selected_features])
    acc = accuracy_score(y_test, preds)
    
    # %99 Doğruluk Ağırlığı + %1 Özellik Azaltma Ağırlığı
    score = (0.99 * acc) + (0.01 * (1 - len(selected_features) / num_features))
    
    cache[key] = score
    return score

# ============================================================
# 4. MFO ANA DÖNGÜSÜ VE ERKEN DURDURMA
# ============================================================
start_time = time.time()
stall_count = 0
last_best_score = -1

print(f"\n🚀 Pervaneler Uçuşa Geçiyor: {num_moths} Pervane, Maks {max_iter} Tur (Erken Durdurma: {patience_limit} Tur)")

for t in range(max_iter):
    iter_start = time.time()
    
    # 1. Pervanelerin Uygunluğunu Ölç
    for i in range(num_moths):
        Moth_Positions[i] = np.clip(Moth_Positions[i], 0, 1)
        Moth_Scores[i] = fitness(Moth_Positions[i])
        
    # 2. Ateşleri (Flames) Güncelle
    if t == 0:
        sort_indices = np.argsort(Moth_Scores)[::-1] 
        Flame_Scores = Moth_Scores[sort_indices].copy()
        Flame_Positions = Moth_Positions[sort_indices].copy()
    else:
        double_positions = np.vstack((Flame_Positions, Moth_Positions))
        double_scores = np.concatenate((Flame_Scores, Moth_Scores))
        
        sort_indices = np.argsort(double_scores)[::-1]
        
        Flame_Scores = double_scores[sort_indices][:num_moths].copy()
        Flame_Positions = double_positions[sort_indices][:num_moths].copy()

    best_flame_score = Flame_Scores[0]
    best_flame_pos = Flame_Positions[0]
    selected_count = np.sum(np.where(best_flame_pos >= 0.5, 1, 0))
    
    print(f"  -> Tur {t+1}/{max_iter} | En İyi Ateş Skoru: {best_flame_score:.5f} | Seçilen Özellik: {int(selected_count)} | Süre: {time.time()-iter_start:.1f} sn")

    # --------------------------------------------------------
    # 🌟 ERKEN DURDURMA KONTROLÜ
    # --------------------------------------------------------
    if abs(best_flame_score - last_best_score) < 1e-6:
        stall_count += 1
        print(f"     [Uyarı] En iyi ateş değişmedi. (Sayaç: {stall_count}/{patience_limit})")
    else:
        stall_count = 0  
        last_best_score = best_flame_score

    if stall_count >= patience_limit:
        print("\n🛑 ERKEN DURDURMA TETİKLENDİ!")
        print(f"Sarmal kilitlendi. {patience_limit} turdur skor değişmiyor.")
        break

    # --------------------------------------------------------
    # 3. Pervanelerin Pozisyonunu Güncelle (Sarmal Uçuş)
    # --------------------------------------------------------
    Flame_no = round(num_moths - t * ((num_moths - 1) / max_iter))
    a = -1 + t * (-1 / max_iter) 
    
    for i in range(num_moths):
        for j in range(num_features):
            if i < Flame_no:
                Distance_to_Flame = abs(Flame_Positions[i, j] - Moth_Positions[i, j])
                t_param = (a - 1) * np.random.rand() + 1
                b = 1 
                
                Moth_Positions[i, j] = Distance_to_Flame * math.exp(b * t_param) * math.cos(t_param * 2 * math.pi) + Flame_Positions[i, j]
            else:
                Distance_to_Flame = abs(Flame_Positions[Flame_no - 1, j] - Moth_Positions[i, j])
                t_param = (a - 1) * np.random.rand() + 1
                b = 1
                
                Moth_Positions[i, j] = Distance_to_Flame * math.exp(b * t_param) * math.cos(t_param * 2 * math.pi) + Flame_Positions[Flame_no - 1, j]

            # Binary Uzaya Çevirme (Sigmoid)
            sigmoid_val = 1 / (1 + np.exp(-10 * (Moth_Positions[i, j] - 0.5)))
            Moth_Positions[i, j] = 1.0 if sigmoid_val >= np.random.rand() else 0.0

# ============================================================
# 5. ALTIN ÖZELLİKLERİ KAYDET
# ============================================================
best_features_indices = np.where(Flame_Positions[0] >= 0.5)[0]

save_path = os.path.join(work_dir, 'MFO_Best_Features_Indices.npy')
np.save(save_path, best_features_indices)

print("\n==================================================")
print("🎯 MFO OPERASYONU TAMAMLANDI!")
print(f"Toplam Süre: {(time.time() - start_time) / 60:.2f} dakika")
print(f"Orijinal 2048 özellikten, {len(best_features_indices)} özellik seçildi.")
print(f"Pervanelerin yandığı bu altın indeksler şuraya kaydedildi:\n{save_path}")
print("==================================================")