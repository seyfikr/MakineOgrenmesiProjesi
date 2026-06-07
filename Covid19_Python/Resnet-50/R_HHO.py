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
print("🦅 HHO (HARRIS HAWKS) RESNET-50 ÖZELLİK SEÇİMİ 🦅")
print("==================================================")

print("\nResNet-50 verileri diskten okunuyor...")
X_all = np.load(os.path.join(work_dir, 'X_features_resnet50.npy'))
y_all = np.load(os.path.join(work_dir, 'y_labels.npy'))

print(f"Toplam Veri: {X_all.shape[0]} Resim | Özellik Sayısı (Uzay): {X_all.shape[1]}")

# Adil Veri Bölme (Orijinal Uyum)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=42, stratify=y_all)

# ============================================================
# 2. HHO PARAMETRELERİ (10 Şahin, 10 Tur, 4 Erken Durdurma)
# ============================================================
num_hawks = 10
max_iter = 10
patience_limit = 4  
num_features = X_train.shape[1]

# Tavşan (Av) - HHO'da en iyi çözüm "Tavşan" olarak adlandırılır
Rabbit_Location = np.zeros(num_features)
Rabbit_Energy_Score = float('-inf')

# Şahinlerin rastgele başlatılması
Hawks_Positions = np.random.randint(2, size=(num_hawks, num_features)).astype(float)
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
# 4. HHO ANA DÖNGÜSÜ VE ERKEN DURDURMA
# ============================================================
start_time = time.time()
stall_count = 0
last_rabbit_score = -1

print(f"\n🚀 Şahinler Havalanıyor: {num_hawks} Şahin, Maks {max_iter} Tur (Erken Durdurma: {patience_limit} Tur)")

for t in range(max_iter):
    iter_start = time.time()
    
    # 1. Şahinlerin Uygunluğunu Ölç ve Avı Bul
    for i in range(num_hawks):
        fit = fitness(Hawks_Positions[i])
        
        if fit > Rabbit_Energy_Score:
            Rabbit_Energy_Score = fit
            Rabbit_Location = Hawks_Positions[i].copy()

    selected_count = np.sum(np.where(Rabbit_Location >= 0.5, 1, 0))
    print(f"  -> Tur {t+1}/{max_iter} | En İyi Skor: {Rabbit_Energy_Score:.5f} | Seçilen Özellik: {int(selected_count)} | Süre: {time.time()-iter_start:.1f} sn")

    # --------------------------------------------------------
    # 🌟 ERKEN DURDURMA KONTROLÜ
    # --------------------------------------------------------
    if abs(Rabbit_Energy_Score - last_rabbit_score) < 1e-6:
        stall_count += 1
        print(f"     [Uyarı] Şahinler tıkandı. (Sayaç: {stall_count}/{patience_limit})")
    else:
        stall_count = 0  
        last_rabbit_score = Rabbit_Energy_Score

    if stall_count >= patience_limit:
        print("\n🛑 ERKEN DURDURMA TETİKLENDİ!")
        print(f"Av tamamen kuşatıldı. {patience_limit} turdur skor değişmiyor.")
        break

    # --------------------------------------------------------
    # 2. Şahinlerin Pozisyonunu Güncelle (Taktiksel Avlanma)
    # --------------------------------------------------------
    E0 = 2 * np.random.rand(num_hawks) - 1  # Başlangıç enerjisi
    E = 2 * E0 * (1 - (t / max_iter))       # Kaçış enerjisi
    
    mean_position = np.mean(Hawks_Positions, axis=0)

    for i in range(num_hawks):
        jump_strength = 2 * (1 - np.random.rand()) 
        
        # Keşif Evresi (Exploration)
        if abs(E[i]) >= 1:
            q = np.random.rand()
            if q >= 0.5:
                rand_hawk_idx = math.floor(num_hawks * np.random.rand())
                rand_hawk = Hawks_Positions[rand_hawk_idx]
                Hawks_Positions[i] = rand_hawk - np.random.rand() * abs(rand_hawk - 2 * np.random.rand() * Hawks_Positions[i])
            else:
                Hawks_Positions[i] = (Rabbit_Location - mean_position) - np.random.rand() * (np.random.rand() * 2 - 1)
        
        # Sömürü Evresi (Exploitation)
        else:
            r = np.random.rand() 
            
            # Yumuşak Kuşatma
            if r >= 0.5 and abs(E[i]) >= 0.5:
                delta_X = Rabbit_Location - Hawks_Positions[i]
                Hawks_Positions[i] = delta_X - E[i] * abs(jump_strength * Rabbit_Location - Hawks_Positions[i])
            # Sert Kuşatma
            elif r >= 0.5 and abs(E[i]) < 0.5:
                delta_X = Rabbit_Location - Hawks_Positions[i]
                Hawks_Positions[i] = Rabbit_Location - E[i] * abs(delta_X)
            # Ani Dalış ile Yumuşak/Sert Kuşatma
            else:
                Hawks_Positions[i] = Rabbit_Location - E[i] * abs(jump_strength * Rabbit_Location - mean_position)

        # Değerleri 0-1 arasına sıkıştırıp Binary (0 veya 1) yapıyoruz
        Hawks_Positions[i] = 1 / (1 + np.exp(-10 * (Hawks_Positions[i] - 0.5)))
        Hawks_Positions[i] = np.where(Hawks_Positions[i] >= np.random.rand(num_features), 1.0, 0.0)

# ============================================================
# 5. ALTIN ÖZELLİKLERİ KAYDET
# ============================================================
best_features_indices = np.where(Rabbit_Location >= 0.5)[0]

save_path = os.path.join(work_dir, 'HHO_Best_Features_Indices.npy')
np.save(save_path, best_features_indices)

print("\n==================================================")
print("🎯 HHO OPERASYONU TAMAMLANDI!")
print(f"Toplam Süre: {(time.time() - start_time) / 60:.2f} dakika")
print(f"Orijinal 2048 özellikten, {len(best_features_indices)} özellik seçildi.")
print(f"Şahinlerin bulduğu bu indeksler şuraya kaydedildi:\n{save_path}")
print("==================================================")