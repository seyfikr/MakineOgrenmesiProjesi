import os
import time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. KLASÖR YOLLARI VE VERİ YÜKLEME
# ============================================================
work_dir = r"C:\Users\ThinkPad\Desktop\Makine\Covid19-3\Densenet201_feature_Selection"

print("==================================================")
print("🐺 GWO (GRİ KURT) ÖZELLİK SEÇİMİ BAŞLIYOR 🐺")
print("==================================================")

# DenseNet-201 özelliklerini yükle
print("\nVeriler diskten okunuyor...")
X_all = np.load(os.path.join(work_dir, 'X_features_densenet201.npy'))
y_all = np.load(os.path.join(work_dir, 'y_labels.npy'))

print(f"Toplam Veri: {X_all.shape[0]} Resim | Özellik Sayısı (Uzay): {X_all.shape[1]}")

# Karargahtaki bu aşamada veriyi bölüyoruz
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=42, stratify=y_all)

# ============================================================
# 2. GWO PARAMETRELERİ (Komutuna Göre Ayarlandı)
# ============================================================
num_wolves =10
max_iter = 10
patience_limit = 4  # Üst üste aynı skor gelme sınırı
num_features = X_train.shape[1]

# Lider Kurtlar
Alpha_pos = np.zeros(num_features)
Alpha_score = float('-inf')

Beta_pos = np.zeros(num_features)
Beta_score = float('-inf')

Delta_pos = np.zeros(num_features)
Delta_score = float('-inf')

# Sürünün rastgele başlatılması (0 ve 1'lerden oluşan matris)
Wolf_Positions = np.random.randint(2, size=(num_wolves, num_features))

# Skor hafızası (Aynı konumu tekrar hesaplamamak için hızlandırıcı)
cache = {}

# ============================================================
# 3. FİTNESS (Uygunluk) FONKSİYONU
# ============================================================
def fitness(position):
    key = tuple(position)
    if key in cache: 
        return cache[key]
    
    selected_features = np.where(position >= 0.5)[0]
    
    # Hiç özellik seçilmezse en kötü skoru ver
    if len(selected_features) == 0: 
        return 0.0
    
    # Hızlı Karar Verici: Eğitim sırasında süreyi kısaltmak için hafif XGBoost kullanıyoruz
    clf = xgb.XGBClassifier(n_estimators=20, max_depth=3, learning_rate=0.1, 
                            tree_method='hist', random_state=42, n_jobs=-1)
    
    clf.fit(X_train[:, selected_features], y_train)
    preds = clf.predict(X_test[:, selected_features])
    acc = accuracy_score(y_test, preds)
    
    # Skorlama: %99 Doğruluk + %1 Özellik Düşürme Başarısı (Mümkün olan en az özellikle en yüksek skoru bulmak için)
    score = (0.99 * acc) + (0.01 * (1 - len(selected_features) / num_features))
    
    cache[key] = score
    return score

# ============================================================
# 4. GWO ANA DÖNGÜSÜ VE ERKEN DURDURMA (EARLY STOPPING)
# ============================================================
start_time = time.time()
stall_count = 0
last_alpha_score = -1

print(f"\n🚀 Sürü Ava Çıkıyor: {num_wolves} Kurt, Maks {max_iter} Tur (Erken Durdurma: {patience_limit} Tur)")

for t in range(max_iter):
    iter_start = time.time()
    
    # 1. Sürünün Uygunluğunu Ölç (Kurtların bulduğu avların kalitesini değerlendir)
    for i in range(num_wolves):
        fit = fitness(Wolf_Positions[i])
        
        # Liderleri Güncelle
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
    # 🌟 ERKEN DURDURMA KONTROLÜ (Senin kuralın)
    # --------------------------------------------------------
    # Eğer Alfa skoru bir önceki turun aynısıysa sayacı artır
    if abs(Alpha_score - last_alpha_score) < 1e-6:
        stall_count += 1
        print(f"     [Uyarı] Skor değişmedi. (Sayaç: {stall_count}/{patience_limit})")
    else:
        stall_count = 0  # Yeni bir rekor kırıldıysa sayacı sıfırla
        last_alpha_score = Alpha_score

    # Eğer 4 tur boyunca skor aynı kaldıysa döngüyü kır!
    if stall_count >= patience_limit:
        print("\n🛑 ERKEN DURDURMA TETİKLENDİ!")
        print(f"Lider Kurt {patience_limit} turdur daha iyi bir av bulamadı. Maksimum verime ulaşıldı.")
        break

    # --------------------------------------------------------
    # 2. Kurtların Pozisyonunu Güncelle (Matematiksel Avlanma)
    # --------------------------------------------------------
    a = 2 - t * (2 / max_iter)  # Avlanma çemberi giderek daralır
    
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

            # Yeni pozisyonu hesapla ve Binary (0 veya 1) uzaya çevir (Sigmoid ile)
            new_pos = (X1 + X2 + X3) / 3
            sigmoid_val = 1 / (1 + np.exp(-10 * (new_pos - 0.5))) # Keskin geçiş
            
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
print(f"Orijinal 1920 özellikten, en keskin {len(best_features_indices)} özellik seçildi.")
print(f"Kurtların bulduğu bu altın indeksler şuraya kaydedildi:\n{save_path}")
print("==================================================")