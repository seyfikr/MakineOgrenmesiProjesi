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
print("🐋 WOA (KAMBUR BALİNA) RESNET-50 ÖZELLİK SEÇİMİ 🐋")
print("==================================================")

print("\nResNet-50 verileri diskten okunuyor...")
X_all = np.load(os.path.join(work_dir, 'X_features_resnet50.npy'))
y_all = np.load(os.path.join(work_dir, 'y_labels.npy'))

print(f"Toplam Veri: {X_all.shape[0]} Resim | Özellik Sayısı (Uzay): {X_all.shape[1]}")

# Adil Veri Bölme (Orijinal Uyum)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=42, stratify=y_all)

# ============================================================
# 2. WOA PARAMETRELERİ (10 Balina, 10 Tur, 4 Erken Durdurma)
# ============================================================
num_whales = 10
max_iter = 10
patience_limit = 4  
num_features = X_train.shape[1]

# Lider Balina (Avı ilk fark eden)
Leader_pos = np.zeros(num_features)
Leader_score = float('-inf')

# Sürünün rastgele başlatılması
Whale_Positions = np.random.randint(2, size=(num_whales, num_features)).astype(float)
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
# 4. WOA ANA DÖNGÜSÜ VE ERKEN DURDURMA
# ============================================================
start_time = time.time()
stall_count = 0
last_leader_score = -1

print(f"\n🚀 Sürü Dalışa Geçiyor: {num_whales} Balina, Maks {max_iter} Tur (Erken Durdurma: {patience_limit} Tur)")

for t in range(max_iter):
    iter_start = time.time()
    
    # 1. Sürünün Uygunluğunu Ölç ve Lideri Güncelle
    for i in range(num_whales):
        fit = fitness(Whale_Positions[i])
        
        if fit > Leader_score:
            Leader_score = fit
            Leader_pos = Whale_Positions[i].copy()

    selected_count = np.sum(np.where(Leader_pos >= 0.5, 1, 0))
    print(f"  -> Tur {t+1}/{max_iter} | En İyi Skor: {Leader_score:.5f} | Seçilen Özellik: {int(selected_count)} | Süre: {time.time()-iter_start:.1f} sn")

    # --------------------------------------------------------
    # 🌟 ERKEN DURDURMA KONTROLÜ
    # --------------------------------------------------------
    if abs(Leader_score - last_leader_score) < 1e-6:
        stall_count += 1
        print(f"     [Uyarı] Lider balina tıkandı. (Sayaç: {stall_count}/{patience_limit})")
    else:
        stall_count = 0  
        last_leader_score = Leader_score

    if stall_count >= patience_limit:
        print("\n🛑 ERKEN DURDURMA TETİKLENDİ!")
        print(f"Sarmal ağ kapandı. {patience_limit} turdur skor değişmiyor.")
        break

    # --------------------------------------------------------
    # 2. Balinaların Pozisyonunu Güncelle (Sarmal Avlanma)
    # --------------------------------------------------------
    a = 2 - t * (2 / max_iter)  
    a2 = -1 + t * (-1 / max_iter) 
    
    for i in range(num_whales):
        r1 = np.random.rand()
        r2 = np.random.rand()
        
        A = 2 * a * r1 - a  
        C = 2 * r2          
        
        b = 1  
        l = (a2 - 1) * np.random.rand() + 1
        
        p = np.random.rand() 
        
        for j in range(num_features):
            if p < 0.5:
                # Rastgele Arama Yap (Arama Uzayını Genişlet)
                if abs(A) >= 1:
                    rand_idx = math.floor(num_whales * np.random.rand())
                    X_rand = Whale_Positions[rand_idx]
                    D_X_rand = abs(C * X_rand[j] - Whale_Positions[i, j])
                    new_pos = X_rand[j] - A * D_X_rand
                # Lidere Yaklaş (Avı Daralt)
                else:
                    D_Leader = abs(C * Leader_pos[j] - Whale_Positions[i, j])
                    new_pos = Leader_pos[j] - A * D_Leader
            # Sarmal Hareket (Avın Etrafında Dön)
            else:
                distance2Leader = abs(Leader_pos[j] - Whale_Positions[i, j])
                new_pos = distance2Leader * math.exp(b * l) * math.cos(l * 2 * math.pi) + Leader_pos[j]
            
            # Değerleri yumuşat ve Binary formata (0 veya 1) çevir
            sigmoid_val = 1 / (1 + np.exp(-10 * (new_pos - 0.5)))
            Whale_Positions[i, j] = 1.0 if sigmoid_val >= np.random.rand() else 0.0

# ============================================================
# 5. ALTIN ÖZELLİKLERİ KAYDET
# ============================================================
best_features_indices = np.where(Leader_pos >= 0.5)[0]

save_path = os.path.join(work_dir, 'WOA_Best_Features_Indices.npy')
np.save(save_path, best_features_indices)

print("\n==================================================")
print("🎯 WOA OPERASYONU TAMAMLANDI!")
print(f"Toplam Süre: {(time.time() - start_time) / 60:.2f} dakika")
print(f"Orijinal 2048 özellikten, {len(best_features_indices)} özellik seçildi.")
print(f"Balinaların bulduğu bu indeksler şuraya kaydedildi:\n{save_path}")
print("==================================================")