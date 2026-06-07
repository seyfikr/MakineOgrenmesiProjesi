import os
import time
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. KLASÖR YOLLARI VE YENİ UZMAN VERİLERİN YÜKLENMESİ
# ============================================================
work_dir = r"C:\Users\ThinkPad\Desktop\Makine\Covid19-3\Densenet201_feature_Selection"

print("==================================================================")
print("🏆 BÜYÜK FİNAL: 5-FOLD CV + SIZINTISIZ PSO + DENSENET + LIGHTGBM 🏆")
print("==================================================================")

print("\nUzman (Fine-Tuned) özellikler diskten okunuyor...")
X_all = np.load(os.path.join(work_dir, 'X_features_densenet201_pytorch_finetuned.npy'))
y_all = np.load(os.path.join(work_dir, 'y_labels_pytorch_finetuned.npy'))
classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

print(f"Toplam Veri: {X_all.shape[0]} Resim | Özellik Sayısı: {X_all.shape[1]}")

# ============================================================
# 2. ŞAMPİYONLAR LİGİ BÖLÜNMESİ (%85 CV İÇİN, %15 TEST İÇİN)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.15, random_state=42, stratify=y_all)

print(f"📚 Eğitim ve CV (Train) : {X_train.shape[0]} resim (%85)")
print(f"🎓 Kilitli Test (Test)  : {X_test.shape[0]} resim (%15)\n")

# ============================================================
# 3. PSO PARAMETRELERİ
# ============================================================
num_particles = 25
max_iter = 20
patience_limit = 4
num_features = X_train.shape[1]

Particle_Positions = np.random.randint(2, size=(num_particles, num_features)).astype(float)
Particle_Velocities = np.zeros((num_particles, num_features))
Pbest_Positions = Particle_Positions.copy()
Pbest_Scores = np.zeros(num_particles)
Gbest_Position = np.zeros(num_features)
Gbest_Score = float('-inf')

cache = {}

# ============================================================
# 4. FİTNESS FONKSİYONU (5-FOLD STRATIFIED CV İLE)
# ============================================================
def fitness(position):
    bin_pos = np.where(position >= 0.5, 1, 0)
    key = tuple(bin_pos)
    if key in cache: return cache[key]
    
    selected_features = np.where(bin_pos == 1)[0]
    if len(selected_features) == 0: return 0.0
    
    X_subset = X_train[:, selected_features]
    clf = lgb.LGBMClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    
    # 5-Fold Cross Validation Tanımlaması
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    
    for train_idx, val_idx in skf.split(X_subset, y_train):
        X_fold_train, X_fold_val = X_subset[train_idx], X_subset[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        clf.fit(X_fold_train, y_fold_train)
        preds = clf.predict(X_fold_val)
        fold_accuracies.append(accuracy_score(y_fold_val, preds))
        
    # 5 Katmanın Ortalamasını Alıyoruz
    mean_acc = np.mean(fold_accuracies)
    
    # Skor Hesaplama: %99 Ortalama Doğruluk, %1 Özellik Azaltma Teşviki
    score = (0.99 * mean_acc) + (0.01 * (1 - len(selected_features) / num_features))
    cache[key] = score
    return score

# ============================================================
# 5. PSO ANA DÖNGÜSÜ
# ============================================================
start_time = time.time()
stall_count = 0
last_gbest_score = -1

print(f"🦅 Sürü Havalanıyor: {num_particles} Kuş, Maksimum {max_iter} Tur (Hedef: 5-Fold CV Maksimizasyonu)")

try:
    for t in range(max_iter):
        iter_start = time.time()
        for i in range(num_particles):
            fit = fitness(Particle_Positions[i])
            if fit > Pbest_Scores[i]:
                Pbest_Scores[i] = fit
                Pbest_Positions[i] = Particle_Positions[i].copy()
            if fit > Gbest_Score:
                Gbest_Score = fit
                Gbest_Position = Particle_Positions[i].copy()

        selected_count = np.sum(np.where(Gbest_Position >= 0.5, 1, 0))
        print(f"  -> Tur {t+1:02d}/{max_iter} | 5-Fold CV Skoru: {Gbest_Score:.5f} | Seçilen Özellik: {int(selected_count)} | Süre: {time.time()-iter_start:.1f} sn")

        if abs(Gbest_Score - last_gbest_score) < 1e-6:
            stall_count += 1
        else:
            stall_count = 0  
            last_gbest_score = Gbest_Score

        if stall_count >= patience_limit:
            print(f"\n✅ OTOMATİK DURDURMA: {patience_limit} turdur 5-Fold skoru değişmedi.")
            break

        w = 0.9 - t * ((0.9 - 0.4) / max_iter)  
        c1, c2 = 2.0, 2.0  
        for i in range(num_particles):
            for j in range(num_features):
                r1, r2 = np.random.rand(), np.random.rand()
                cog_vel = c1 * r1 * (Pbest_Positions[i, j] - Particle_Positions[i, j])
                soc_vel = c2 * r2 * (Gbest_Position[j] - Particle_Positions[i, j])
                Particle_Velocities[i, j] = np.clip(w * Particle_Velocities[i, j] + cog_vel + soc_vel, -6, 6)
                
                sigmoid_v = 1 / (1 + np.exp(-Particle_Velocities[i, j]))
                Particle_Positions[i, j] = 1.0 if np.random.rand() < sigmoid_v else 0.0

except KeyboardInterrupt:
    print("\n🛑 ACİL DURDURMA (CTRL+C)! O ana kadar bulunan en iyi özellikler kaydediliyor...")

# ============================================================
# 6. ALTIN İNDEKSLERİ KAYDET
# ============================================================
best_features_indices = np.where(Gbest_Position >= 0.5)[0]
save_path = os.path.join(work_dir, 'PSO_Best_Features_CV_Finetuned.npy')
np.save(save_path, best_features_indices)

print(f"\n🎯 PSO TAMAMLANDI! {num_features} özellikten CV onaylı en güçlü {len(best_features_indices)} tanesi seçildi.")

# ============================================================
# 7. NİHAİ TEST VE GRAFİKLER (KASADAKİ VERİ AÇILIYOR)
# ============================================================
print("\n==================================================================")
print("🚀 NİHAİ TEST AŞAMASI BAŞLIYOR (KASADAKİ %15 VERİ AÇILIYOR)")
print("==================================================================")

X_train_sel = X_train[:, best_features_indices]
X_test_sel = X_test[:, best_features_indices]

# Seçilen özellikleri kullanarak modeli tüm Eğitim+CV (%85) verisiyle son kez eğitiyoruz
final_clf = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
final_clf.fit(X_train_sel, y_train)

test_preds = final_clf.predict(X_test_sel)
test_preds_prob = final_clf.predict_proba(X_test_sel)

test_acc = accuracy_score(y_test, test_preds)
test_rec = recall_score(y_test, test_preds, average='weighted')
test_f1 = f1_score(y_test, test_preds, average='weighted')

print(f"🌟 NİHAİ ACCURACY : % {test_acc * 100:.2f}")
print(f"🌟 NİHAİ RECALL   : % {test_rec * 100:.2f}")
print(f"🌟 NİHAİ F1-SCORE : % {test_f1 * 100:.2f}")
print("==================================================================\n")

print("Matrisler çiziliyor ve kaydediliyor...")

# Confusion Matrix
cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title(f'Confusion Matrix (CV+PSO)\nAccuracy: %{test_acc*100:.2f}')
plt.ylabel('Gerçek Değerler')
plt.xlabel('Modelin Tahmini')
plt.savefig(os.path.join(work_dir, 'Final_CV_Confusion_Matrix.png'), bbox_inches='tight')
plt.close()

# ROC Curve
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
n_classes = y_test_bin.shape[1]
fpr, tpr, roc_auc = dict(), dict(), dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], test_preds_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'purple']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{classes[i]} (AUC = {roc_auc[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisi (CV+PSO)')
plt.legend(loc="lower right")
plt.savefig(os.path.join(work_dir, 'Final_CV_ROC_Curve.png'), bbox_inches='tight')
plt.close()

print(f"✅ Bütün işlemler kusursuz tamamlandı. Grafikler klasöre kaydedildi!")