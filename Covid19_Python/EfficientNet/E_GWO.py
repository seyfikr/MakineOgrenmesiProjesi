import os
import time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. KLASÖR YOLLARI VE VERİ YÜKLEME (YENİ KARARGAH)
# ============================================================
work_dir = r"C:\Users\ThinkPad\Desktop\Makine\Covid19-3\EfficientNet_feature_Selection"

print("==================================================")
print("🐺 GWO (GRİ KURT) EFFICIENTNET ÖZELLİK SEÇİMİ 🐺")
print("==================================================")

print("\nEfficientNet verileri diskten okunuyor...")
X_all = np.load(os.path.join(work_dir, 'X_features_efficientnet.npy'))
y_all = np.load(os.path.join(work_dir, 'y_labels.npy'))

print(f"Toplam Veri: {X_all.shape[0]} Resim | Özellik Sayısı (Uzay): {X_all.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=42, stratify=y_all)

# ============================================================
# 2. GWO PARAMETRELERİ (8 Kurt, 8 Tur, 4 Erken Durdurma)
# ============================================================
num_wolves = 10
max_iter = 10
patience_limit = 4  
num_features = X_train.shape[1]

Alpha_pos = np.zeros(num_features)
Alpha_score = float('-inf')

Beta_pos = np.zeros(num_features)
Beta_score = float('-inf')

Delta_pos = np.zeros(num_features)
Delta_score = float('-inf')

Wolf_Positions = np.random.randint(2, size=(num_wolves, num_features))
cache = {}

# ============================================================
# 3. FİTNESS (Uygunluk) FONKSİYONU
# ============================================================
def fitness(position):
    key = tuple(position)
    if key in cache: 
        return cache[key]
    
    selected_features = np.where(position >= 0.5)[0]
    
    if len(selected_features) == 0: 
        return 0.0
    
    clf = xgb.XGBClassifier(n_estimators=20, max_depth=3, learning_rate=0.1, 
                            tree_method='hist', random_state=42, n_jobs=-1)
    
    clf.fit(X_train[:, selected_features], y_train)
    preds = clf.predict(X_test[:, selected_features])
    acc = accuracy_score(y_test, preds)
    
    score = (0.99 * acc) + (0.01 * (1 - len(selected_features) / num_features))
    
    cache[key] = score
    return score

# ============================================================
# 4. GWO ANA DÖNGÜSÜ VE ERKEN DURDURMA
# ============================================================
start_time = time.time()
stall_count = 0
last_alpha_score = -1

print(f"\n🚀 Sürü Ava Çıkıyor: {num_wolves} Kurt, Maks {max_iter} Tur (Erken Durdurma: {patience_limit} Tur)")

for t in range(max_iter):
    iter_start = time.time()
    
    for i in range(num_wolves):
        fit = fitness(Wolf_Positions[i])
        
        if fit > Alpha_score:
            Delta_score, Delta_pos = Beta_score, Beta_pos.copy()
            Beta_score, Beta_pos = Alpha_score, Alpha_pos.copy()
            Alpha_score, Alpha_pos = fit, Wolf_Positions[i].copy()
            
        elif fit > Beta_score:
            Delta_score, Delta_pos = Beta_score, Beta_pos.copy()
            Beta_score, Beta_pos = fit, Wolf_Positions[i].copy()
            
        elif fit > Delta_score:
            Delta_score, Delta_pos = fit, Wolf_Positions[i].copy()

    selected_count = np.sum(Alpha_pos)
    print(f"  -> Tur {t+1}/{max_iter} | Alfa Skoru: {Alpha_score:.5f} | Seçilen Özellik: {int(selected_count)} | Süre: {time.time()-iter_start:.1f} sn")

    # --------------------------------------------------------
    # 🌟 ERKEN DURDURMA KONTROLÜ
    # --------------------------------------------------------
    if abs(Alpha_score - last_alpha_score) < 1e-6:
        stall_count += 1
        print(f"     [Uyarı] Skor değişmedi. (Sayaç: {stall_count}/{patience_limit})")
    else:
        stall_count = 0  
        last_alpha_score = Alpha_score

    if stall_count >= patience_limit:
        print("\n🛑 ERKEN DURDURMA TETİKLENDİ!")
        print(f"Lider Kurt {patience_limit} turdur daha iyi bir av bulamadı.")
        break

    # --------------------------------------------------------
    # 2. Kurtların Pozisyonunu Güncelle
    # --------------------------------------------------------
    a = 2 - t * (2 / max_iter)  
    
    for i in range(num_wolves):
        for j in range(num_features):
            r1, r2 = np.random.rand(), np.random.rand()
            A1 = 2 * a * r1 - a
            X1 = Alpha_pos[j] - A1 * abs(2 * r2 * Alpha_pos[j] - Wolf_Positions[i, j])

            r1, r2 = np.random.rand(), np.random.rand()
            A2 = 2 * a * r1 - a
            X2 = Beta_pos[j] - A2 * abs(2 * r2 * Beta_pos[j] - Wolf_Positions[i, j])

            r1, r2 = np.random.rand(), np.random.rand()
            A3 = 2 * a * r1 - a
            X3 = Delta_pos[j] - A3 * abs(2 * r2 * Delta_pos[j] - Wolf_Positions[i, j])

            new_pos = (X1 + X2 + X3) / 3
            sigmoid_val = 1 / (1 + np.exp(-10 * (new_pos - 0.5))) 
            
            Wolf_Positions[i, j] = 1 if sigmoid_val >= np.random.rand() else 0

# ============================================================
# 5. ALTIN ÖZELLİKLERİ KAYDET
# ============================================================
best_features_indices = np.where(Alpha_pos >= 0.5)[0]

save_path = os.path.join(work_dir, 'GWO_Best_Features_Indices.npy')
np.save(save_path, best_features_indices)

print("\n==================================================")
print("🎯 GWO OPERASYONU TAMAMLANDI!")
print(f"Toplam Süre: {(time.time() - start_time) / 60:.2f} dakika")
print(f"Orijinal 1280 özellikten, en keskin {len(best_features_indices)} özellik seçildi.")
print(f"Kurtların bulduğu bu altın indeksler şuraya kaydedildi:\n{save_path}")
print("==================================================")