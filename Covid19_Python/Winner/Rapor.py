import os
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. VERİLERİN YÜKLENMESİ VE HAZIRLIK
# ============================================================
work_dir = r"C:\Users\ThinkPad\Desktop\Makine\Covid19-3\Densenet201_feature_Selection"
classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

# Özellikleri ve PSO ile seçilen altın indeksleri yükle
X_all = np.load(os.path.join(work_dir, 'X_features_densenet201_pytorch_finetuned.npy'))
y_all = np.load(os.path.join(work_dir, 'y_labels_pytorch_finetuned.npy'))
best_features = np.load(os.path.join(work_dir, 'PSO_Best_Features_Finetuned.npy'))

# Sadece seçilen özellikleri al
X_selected = X_all[:, best_features]

# ============================================================
# 2. STANDART (70-15-15) EĞİTİM VE SINIF BAZLI RAPORLAR
# ============================================================
X_temp, X_test, y_temp, y_test = train_test_split(X_selected, y_all, test_size=0.15, random_state=42, stratify=y_all)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.15/0.85), random_state=42, stratify=y_temp)

clf_standard = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
clf_standard.fit(X_train, y_train)

print("\n" + "="*70)
print("📚 1. EĞİTİM SETİ (TRAIN) SINIF BAZLI RAPORU (%70 Veri)")
print("="*70)
print(classification_report(y_train, clf_standard.predict(X_train), target_names=classes, digits=4))

print("\n" + "="*70)
print("📝 2. DOĞRULAMA SETİ (VALIDATION) SINIF BAZLI RAPORU (%15 Veri)")
print("="*70)
print(classification_report(y_val, clf_standard.predict(X_val), target_names=classes, digits=4))

print("\n" + "="*70)
print("🎓 3. KİLİTLİ TEST SETİ (TEST) SINIF BAZLI RAPORU (%15 Veri)")
print("="*70)
print(classification_report(y_test, clf_standard.predict(X_test), target_names=classes, digits=4))

# ============================================================
# 3. 10-FOLD CROSS VALIDATION (TÜM VERİ ÜZERİNDE)
# ============================================================
print("\n" + "="*70)
print("🔄 4. ALTIN STANDART: 10-FOLD CROSS VALIDATION BAŞLIYOR 🔄")
print("="*70)

# Veriyi 10 eşit ve dengeli parçaya böl
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
acc_scores = []
fold = 1

for train_idx, test_idx in skf.split(X_selected, y_all):
    # O tur için eğitim ve test verilerini ayır
    X_tr, X_te = X_selected[train_idx], X_selected[test_idx]
    y_tr, y_te = y_all[train_idx], y_all[test_idx]
    
    # Yeni model oluştur ve eğit
    clf_cv = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    clf_cv.fit(X_tr, y_tr)
    
    # Test et ve skoru kaydet
    acc = accuracy_score(y_te, clf_cv.predict(X_te))
    acc_scores.append(acc)
    
    print(f"  -> Fold {fold:02d}/10 | Doğruluk (Accuracy): % {acc * 100:.2f}")
    fold += 1

print("-" * 70)
print(f"🌟 10-FOLD ORTALAMA ACCURACY: % {np.mean(acc_scores) * 100:.2f} (± % {np.std(acc_scores) * 100:.2f} sapma)")
print("="*70)