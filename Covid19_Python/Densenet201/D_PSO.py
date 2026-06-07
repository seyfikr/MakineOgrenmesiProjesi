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
print("🐦 PSO (KUŞ SÜRÜSÜ) ÖZELLİK SEÇİMİ BAŞLIYOR 🐦")
print("==================================================")

print("\nVeriler diskten okunuyor...")
X_all = np.load(os.path.join(work_dir, 'X_features_densenet201.npy'))
y_all = np.load(os.path.join(work_dir, 'y_labels.npy'))

print(f"Toplam Veri: {X_all.shape[0]} Resim | Özellik Sayısı (Uzay): {X_all.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=42, stratify=y_all)

# ============================================================
# 2. PSO PARAMETRELERİ
# ============================================================
num_particles = 10
max_iter = 10
patience_limit = 4  
num_features = X_train.shape[1]

# Kuşların Başlangıç Pozisyonları (0 ve 1'ler) ve Hızları
Particle_Positions = np.random.randint(2, size=(num_particles, num_features)).astype(float)
Particle_Velocities = np.zeros((num_particles, num_features))

# Bireysel En İyi (Personal Best - pbest)
Pbest_Positions = Particle_Positions.copy()
Pbest_Scores = np.zeros(num_particles)

# Küresel En İyi (Global Best - gbest)
Gbest_Position = np.zeros(num_features)
Gbest_Score = float('-inf')

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
    
    clf = xgb.XGBClassifier(n_estimators=20, max_depth=3, learning_rate=0.1, 
                            tree_method='hist', random_state=42, n_jobs=-1)
    
    clf.fit(X_train[:, selected_features], y_train)
    preds = clf.predict(X_test[:, selected_features])
    acc = accuracy_score(y_test, preds)
    
    score = (0.99 * acc) + (0.01 * (1 - len(selected_features) / num_features))
    
    cache[key] = score
    return score

# ============================================================
# 4. PSO ANA DÖNGÜSÜ VE ERKEN DURDURMA
# ============================================================
start_time = time.time()
stall_count = 0
last_gbest_score = -1

print(f"\n🚀 Sürü Havalanıyor: {num_particles} Kuş, Maks {max_iter} Tur (Erken Durdurma: {patience_limit} Tur)")

for t in range(max_iter):
    iter_start = time.time()
    
    # 1. Kuşların Uygunluğunu Ölç ve Hafızaları Güncelle
    for i in range(num_particles):
        fit = fitness(Particle_Positions[i])
        
        # Bireysel Hafızayı Güncelle (Pbest)
        if fit > Pbest_Scores[i]:
            Pbest_Scores[i] = fit
            Pbest_Positions[i] = Particle_Positions[i].copy()
            
        # Sürünün Ortak Hafızasını Güncelle (Gbest)
        if fit > Gbest_Score:
            Gbest_Score = fit
            Gbest_Position = Particle_Positions[i].copy()

    selected_count = np.sum(np.where(Gbest_Position >= 0.5, 1, 0))
    print(f"  -> Tur {t+1}/{max_iter} | En İyi Skor: {Gbest_Score:.5f} | Seçilen Özellik: {int(selected_count)} | Süre: {time.time()-iter_start:.1f} sn")

    # --------------------------------------------------------
    # 🌟 ERKEN DURDURMA KONTROLÜ
    # --------------------------------------------------------
    if abs(Gbest_Score - last_gbest_score) < 1e-6:
        stall_count += 1
        print(f"     [Uyarı] Sürü aynı yere yığıldı. (Sayaç: {stall_count}/{patience_limit})")
    else:
        stall_count = 0  
        last_gbest_score = Gbest_Score

    if stall_count >= patience_limit:
        print("\n🛑 ERKEN DURDURMA TETİKLENDİ!")
        print(f"Sürü yerel optimuma kilitlendi. {patience_limit} turdur skor değişmiyor.")
        break

    # --------------------------------------------------------
    # 2. Kuşların Hızını ve Pozisyonunu Güncelle
    # --------------------------------------------------------
    w = 0.9 - t * ((0.9 - 0.4) / max_iter)  # Atalet (Inertia) - Zamanla yavaşlarlar
    c1 = 2.0  # Bilişsel Katsayı (Kendi hafızasına güven)
    c2 = 2.0  # Sosyal Katsayı (Sürüye güven)
    
    for i in range(num_particles):
        for j in range(num_features):
            r1, r2 = np.random.rand(), np.random.rand()
            
            # Hız Güncellemesi
            cognitive_vel = c1 * r1 * (Pbest_Positions[i, j] - Particle_Positions[i, j])
            social_vel = c2 * r2 * (Gbest_Position[j] - Particle_Positions[i, j])
            Particle_Velocities[i, j] = w * Particle_Velocities[i, j] + cognitive_vel + social_vel
            
            # Hızları sınırla (Çok uzağa uçmalarını engelle)
            Particle_Velocities[i, j] = np.clip(Particle_Velocities[i, j], -6, 6)
            
            # Pozisyon Güncellemesi (Binary PSO için Sigmoid)
            sigmoid_v = 1 / (1 + np.exp(-Particle_Velocities[i, j]))
            
            if np.random.rand() < sigmoid_v:
                Particle_Positions[i, j] = 1.0
            else:
                Particle_Positions[i, j] = 0.0

# ============================================================
# 5. ALTIN ÖZELLİKLERİ KAYDET
# ============================================================
best_features_indices = np.where(Gbest_Position >= 0.5)[0]

save_path = os.path.join(work_dir, 'PSO_Best_Features_Indices.npy')
np.save(save_path, best_features_indices)

print("\n==================================================")
print("🎯 PSO OPERASYONU TAMAMLANDI!")
print(f"Toplam Süre: {(time.time() - start_time) / 60:.2f} dakika")
print(f"Orijinal 1920 özellikten, {len(best_features_indices)} özellik seçildi.")
print(f"Kuşların bulduğu bu indeksler şuraya kaydedildi:\n{save_path}")
print("==================================================")