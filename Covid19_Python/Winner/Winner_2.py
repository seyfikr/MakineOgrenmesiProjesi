import os
import time
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score, f1_score, precision_score, 
                             confusion_matrix, roc_curve, auc, precision_recall_curve, 
                             average_precision_score, mean_squared_error, 
                             mean_absolute_error, r2_score)
from sklearn.preprocessing import label_binarize
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. KLASÖR YOLLARI VE YENİ UZMAN VERİLERİN YÜKLENMESİ
# ============================================================
work_dir = r"C:\Users\ThinkPad\Desktop\Makine\Covid19-3\Densenet201_feature_Selection"

# YENİ KLASÖR OLUŞTURULUYOR
report_dir = os.path.join(work_dir, "Detayli_Analiz_Raporlari")
os.makedirs(report_dir, exist_ok=True)

print("==================================================================")
print("🏆 BÜYÜK FİNAL: PSO OPTİMİZASYONU VE DETAYLI ANALİZ SİSTEMİ 🏆")
print("==================================================================")
print(f"📁 Tüm grafikler ve sonuçlar '{report_dir}' klasörüne kaydedilecek.\n")

print("Uzman (Fine-Tuned) özellikler diskten okunuyor...")
X_all = np.load(os.path.join(work_dir, 'X_features_densenet201_pytorch_finetuned.npy'))
y_all = np.load(os.path.join(work_dir, 'y_labels_pytorch_finetuned.npy'))
classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

print(f"Toplam Veri: {X_all.shape[0]} Resim | Başlangıç Özellik Sayısı: {X_all.shape[1]}\n")

# ============================================================
# 2. ŞAMPİYONLAR LİGİ BÖLÜNMESİ (70-15-15)
# ============================================================
X_temp, X_test, y_temp, y_test = train_test_split(X_all, y_all, test_size=0.15, random_state=42, stratify=y_all)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.15/0.85), random_state=42, stratify=y_temp)

print(f"📚 Eğitim (Train)     : {X_train.shape[0]} resim")
print(f"📝 Doğrulama (Val)    : {X_val.shape[0]} resim")
print(f"🎓 Kilitli Test (Test): {X_test.shape[0]} resim\n")

# ============================================================
# 3. PSO (PARÇACIK SÜRÜ OPTİMİZASYONU) ANA DÖNGÜSÜ
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

def fitness(position):
    bin_pos = np.where(position >= 0.5, 1, 0)
    key = tuple(bin_pos)
    if key in cache: return cache[key]
    
    selected_features = np.where(bin_pos == 1)[0]
    if len(selected_features) == 0: return 0.0
    
    clf = lgb.LGBMClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    clf.fit(X_train[:, selected_features], y_train)
    preds = clf.predict(X_val[:, selected_features])
    
    acc = accuracy_score(y_val, preds)
    score = (0.99 * acc) + (0.01 * (1 - len(selected_features) / num_features))
    cache[key] = score
    return score

start_time = time.time()
stall_count = 0
last_gbest_score = -1

print(f"🦅 PSO Sürüsü Havalanıyor: {num_particles} Parçacık, Maksimum {max_iter} İterasyon")

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
        print(f"  -> Tur {t+1:02d}/{max_iter} | Doğrulama Skoru: {Gbest_Score:.5f} | Seçilen Özellik: {int(selected_count)} | Süre: {time.time()-iter_start:.1f} sn")

        if abs(Gbest_Score - last_gbest_score) < 1e-6:
            stall_count += 1
        else:
            stall_count = 0  
            last_gbest_score = Gbest_Score

        if stall_count >= patience_limit:
            print(f"\n✅ OTOMATİK DURDURMA: {patience_limit} turdur skor değişmedi.")
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
    print("\n🛑 ACİL DURDURMA! O ana kadar bulunan en iyi özellikler kullanılıyor...")

best_features_indices = np.where(Gbest_Position >= 0.5)[0]
save_path = os.path.join(report_dir, 'PSO_Best_Features_Finetuned.npy')
np.save(save_path, best_features_indices)
print(f"\n🎯 PSO TAMAMLANDI! {num_features} özellikten en güçlü {len(best_features_indices)} tanesi seçildi.")

# ============================================================
# 4. NİHAİ MODELİN EĞİTİMİ (KAYIP VE DOĞRULUK TAKİBİ İLE)
# ============================================================
print("\n==================================================================")
print("🚀 NİHAİ LIGHTGBM TESTİ VE GRAFİK ÇİZİMLERİ BAŞLIYOR")
print("==================================================================")

X_train_sel = X_train[:, best_features_indices]
X_test_sel = X_test[:, best_features_indices]

# Log tutmak için evals_result sözlüğü oluşturuyoruz
evals_result = {}
final_clf = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)

# eval_set içine hem train hem test veriyoruz ki öğrenme/kayıp eğrilerini çizebilelim
final_clf.fit(
    X_train_sel, y_train,
    eval_set=[(X_train_sel, y_train), (X_test_sel, y_test)],
    eval_metric=['multi_error', 'multi_logloss'],
    callbacks=[lgb.record_evaluation(evals_result)]
)

test_preds = final_clf.predict(X_test_sel)
test_preds_prob = final_clf.predict_proba(X_test_sel)

# ============================================================
# 5. METRİKLERİN HESAPLANMASI
# ============================================================
test_acc = accuracy_score(y_test, test_preds)
test_rec = recall_score(y_test, test_preds, average='weighted')
test_prec = precision_score(y_test, test_preds, average='weighted')
test_f1 = f1_score(y_test, test_preds, average='weighted')

# Specificity (Özgünlük) Hesaplaması
cm = confusion_matrix(y_test, test_preds)
specificities = []
for i in range(len(classes)):
    tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
    fp = np.sum(cm[:, i]) - cm[i, i]
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificities.append(spec)
macro_specificity = np.mean(specificities)

# Regresyon Tabanlı Zorunlu Metrikler (Şablon İçin)
mse = mean_squared_error(y_test, test_preds)
mae = mean_absolute_error(y_test, test_preds)
r2 = r2_score(y_test, test_preds)

print(f"\n🔹 Doğruluk (Accuracy)    : % {test_acc * 100:.2f}")
print(f"🔹 Duyarlılık (Recall)    : % {test_rec * 100:.2f}")
print(f"🔹 Kesinlik (Precision)   : % {test_prec * 100:.2f}")
print(f"🔹 F1-Skor                : % {test_f1 * 100:.2f}")
print(f"🔹 Özgünlük (Specificity) : % {macro_specificity * 100:.2f}")
print("-" * 50)
print(f"🔸 Ortalama Kare Hata (MSE)   : {mse:.4f}")
print(f"🔸 Ortalama Mutlak Hata (MAE) : {mae:.4f}")
print(f"🔸 R2 Skoru                   : {r2:.4f}\n")

# ============================================================
# 6. GRAFİKLERİN ÇİZİLMESİ VE KAYDEDİLMESİ
# ============================================================
print("Grafikler oluşturuluyor ve diske yazılıyor...")

# 1. Hata Matrisi
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title(f'Hata Matrisi (Accuracy: %{test_acc*100:.2f})')
plt.ylabel('Gerçek Sınıflar')
plt.xlabel('Tahmin Edilen Sınıflar')
plt.savefig(os.path.join(report_dir, '01_Hata_Matrisi.png'), bbox_inches='tight', dpi=300)
plt.close()

# 2. Kayıp (Loss) Eğrisi
train_loss = evals_result['training']['multi_logloss']
test_loss = evals_result['valid_1']['multi_logloss']
plt.figure(figsize=(8, 5))
plt.plot(train_loss, label='Eğitim Kaybı', lw=2)
plt.plot(test_loss, label='Test Kaybı', lw=2)
plt.title('Eğitim ve Test Setleri İçin Kayıp (Loss) Eğrileri')
plt.xlabel('İterasyon')
plt.ylabel('LogLoss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(report_dir, '02_Kayip_Loss_Egrisi.png'), bbox_inches='tight', dpi=300)
plt.close()

# 3. Doğruluk (Accuracy) Eğrisi
train_acc_curve = [1 - x for x in evals_result['training']['multi_error']]
test_acc_curve = [1 - x for x in evals_result['valid_1']['multi_error']]
plt.figure(figsize=(8, 5))
plt.plot(train_acc_curve, label='Eğitim Doğruluğu', lw=2)
plt.plot(test_acc_curve, label='Test Doğruluğu', lw=2)
plt.title('Eğitim ve Test Setleri İçin Doğruluk Eğrileri')
plt.xlabel('İterasyon')
plt.ylabel('Doğruluk')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(report_dir, '03_Dogruluk_Accuracy_Egrisi.png'), bbox_inches='tight', dpi=300)
plt.close()

# 4. ROC ve AUC Eğrileri
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
colors = ['blue', 'red', 'green', 'purple']
plt.figure(figsize=(8, 6))
for i, color in zip(range(4), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], test_preds_prob[:, i])
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{classes[i]} (AUC = {auc(fpr, tpr):.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Özgünlük)')
plt.ylabel('True Positive Rate (Duyarlılık)')
plt.title('ROC Eğrileri ve AUC Değerleri')
plt.legend(loc="lower right")
plt.savefig(os.path.join(report_dir, '04_ROC_AUC_Egrileri.png'), bbox_inches='tight', dpi=300)
plt.close()

# 5. Precision - Recall Eğrileri
plt.figure(figsize=(8, 6))
for i, color in zip(range(4), colors):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], test_preds_prob[:, i])
    avg_prec = average_precision_score(y_test_bin[:, i], test_preds_prob[:, i])
    plt.plot(recall, precision, color=color, lw=2, label=f'{classes[i]} (AP = {avg_prec:.3f})')
plt.xlabel('Recall (Duyarlılık)')
plt.ylabel('Precision (Kesinlik)')
plt.title('Precision - Recall Eğrileri')
plt.legend(loc="lower left")
plt.savefig(os.path.join(report_dir, '05_Precision_Recall_Egrileri.png'), bbox_inches='tight', dpi=300)
plt.close()

print(f"✅ TÜM İŞLEMLER KUSURSUZ TAMAMLANDI! Sonuçlar klasörde seni bekliyor.")