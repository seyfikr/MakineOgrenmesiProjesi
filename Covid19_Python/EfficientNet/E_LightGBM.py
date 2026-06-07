import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. KLASÖR YOLLARI VE VERİ YÜKLEME (EFFICIENTNET KARARGAHI)
# ============================================================
work_dir = r"C:\Users\ThinkPad\Desktop\Makine\Covid19-3\EfficientNet_feature_Selection"

print("==================================================")
print("⚡ BÜYÜK FİNAL: EFFICIENTNET - 5 ATLI LIGHTGBM ARENASINDA ⚡")
print("==================================================")

print("\nEfficientNet özellikleri (X) ve etiketler (y) yükleniyor...")
X_all = np.load(os.path.join(work_dir, 'X_features_efficientnet.npy'))
y_all = np.load(os.path.join(work_dir, 'y_labels.npy'))
classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

# ============================================================
# 2. ADİL VERİ BÖLME (%80 Eğitim, %20 Test)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=42, stratify=y_all)

print(f"Eğitim Seti (Train) : {X_train.shape[0]} resim")
print(f"Test Seti (Test)    : {X_test.shape[0]} resim\n")

# ============================================================
# 3. YARIŞMACILARI VE LIGHTGBM MOTORUNU HAZIRLAMA
# ============================================================
algorithms = ['GWO', 'HHO', 'WOA', 'MFO', 'PSO']
results = {}

best_algo = ""
best_score = 0
best_model = None
best_X_test = None

print("LightGBM Modelleri Eğitiliyor (Terminali takip et)...\n")
print("-" * 50)

for algo in algorithms:
    index_file = os.path.join(work_dir, f'{algo}_Best_Features_Indices.npy')
    
    if not os.path.exists(index_file):
        print(f"⚠️ {algo} indeks dosyası bulunamadı, atlanıyor...\n")
        print("-" * 50)
        continue
        
    selected_indices = np.load(index_file)
    num_features = len(selected_indices)
    
    X_train_sel = X_train[:, selected_indices]
    X_test_sel = X_test[:, selected_indices]
    
    # LightGBM Modeli Kurulumu (verbose=-1 terminali temiz tutar)
    clf = lgb.LGBMClassifier(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1, 
        random_state=42, 
        n_jobs=-1,
        verbose=-1
    )
    
    print(f"⏳ [{algo}] LightGBM eğitimi başladı... (Seçilen Özellik: {num_features})")
    start_time = time.time()
    
    clf.fit(X_train_sel, y_train)
    
    test_preds = clf.predict(X_test_sel)
    
    # Detaylı Metrikler
    test_acc = accuracy_score(y_test, test_preds)
    test_rec = recall_score(y_test, test_preds, average='weighted')
    test_f1 = f1_score(y_test, test_preds, average='weighted')
    
    train_time = time.time() - start_time
    
    results[algo] = {
        'features': num_features,
        'accuracy': test_acc,
        'recall': test_rec,
        'f1_score': test_f1,
        'time': train_time
    }
    
    print(f"✅ [{algo}] Tamamlandı!")
    print(f"   -> Süre     : {train_time:.2f} saniye")  # LightGBM çok hızlıdır, saniye bazında yazdırıyoruz
    print(f"   -> Accuracy : % {test_acc*100:.2f}")
    print(f"   -> Recall   : % {test_rec*100:.2f}")
    print(f"   -> F1-Score : % {test_f1*100:.2f}")
    print("-" * 50)
    
    # En iyiyi belirle
    if test_acc > best_score:
        best_score = test_acc
        best_algo = algo
        best_model = clf
        best_X_test = X_test_sel

# ============================================================
# 4. KAPIŞMA SONUÇ TABLOSU
# ============================================================
print("\n==========================================================================")
print("🏆 KESİN SONUÇLAR (EFFICIENTNET + LIGHTGBM) 🏆")
print("==========================================================================")
print(f"{'Algoritma':<10} | {'Özellik':<8} | {'Accuracy':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 74)
for algo, data in results.items():
    print(f"{algo:<10} | {str(data['features']):<8} | % {data['accuracy']*100:<8.2f} | % {data['recall']*100:<8.2f} | % {data['f1_score']*100:<8.2f}")
print("==========================================================================")

print(f"\n👑 ŞAMPİYON: {best_algo} (Accuracy: % {best_score*100:.2f})")
print("==========================================================================\n")

# ============================================================
# 5. ŞAMPİYONUN MATRİSLERİNİ ÇİZ VE KAYDET
# ============================================================
if best_model is not None:
    print(f"Şampiyon {best_algo} için Karmaşıklık Matrisi ve ROC Eğrisi Çiziliyor...")
    
    y_pred = best_model.predict(best_X_test)
    y_pred_prob = best_model.predict_proba(best_X_test)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix (Champion: {best_algo} + LightGBM)')
    plt.ylabel('Gerçek Değerler')
    plt.xlabel('Modelin Tahmini')
    cm_path = os.path.join(work_dir, f'Best_{best_algo}_LightGBM_Confusion_Matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    
    # ROC Curve
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    n_classes = y_test_bin.shape[1]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'purple']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{classes[i]} (AUC = {roc_auc[i]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Yanlış Pozitif Oranı)')
    plt.ylabel('True Positive Rate (Doğru Pozitif Oranı)')
    plt.title(f'ROC Eğrisi (Champion: {best_algo} + LightGBM)')
    plt.legend(loc="lower right")
    
    roc_path = os.path.join(work_dir, f'Best_{best_algo}_LightGBM_ROC_Curve.png')
    plt.savefig(roc_path, bbox_inches='tight')
    plt.close()

    print(f"Harika! Grafikler doğrudan şu klasöre kaydedildi:\n{work_dir}")