import os
import cv2
import numpy as np
from glob import glob

# ============================================================
# 1. PATHS
# ============================================================
dataset_dir = r"C:\Users\ThinkPad\Desktop\Makine\Covid19-3\dataset"
output_dir  = r"C:\Users\ThinkPad\Desktop\Makine\Covid19-3\dataset_masked"

classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

# ============================================================
# 2. CLAHE (kontrast iyileştirme - tıbbi görüntüler için kritik)
# ============================================================
def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

# ============================================================
# 3. MASK CLEANING (gürültü temizleme + kenar düzeltme)
# ============================================================
def clean_mask(mask):
    # Binary
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Küçük gürültüleri temizle (opening)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Delikleri doldur (closing)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

# ============================================================
# 4. ROI EXPANSION (çok önemli → context kaybını azaltır)
# ============================================================
def expand_mask(mask, iterations=1):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask, kernel, iterations=iterations)

# ============================================================
# 5. MAIN PIPELINE
# ============================================================
def process_dataset():
    os.makedirs(output_dir, exist_ok=True)

    total = 0

    for cls in classes:
        print(f"\n[{cls}] işleniyor...")

        img_folder = os.path.join(dataset_dir, cls, 'images')
        mask_folder = os.path.join(dataset_dir, cls, 'masks')
        out_folder  = os.path.join(output_dir, cls)

        os.makedirs(out_folder, exist_ok=True)

        image_paths = glob(os.path.join(img_folder, "*.png"))

        for img_path in image_paths:
            name = os.path.basename(img_path)
            mask_path = os.path.join(mask_folder, name)

            # -------------------------
            # 1. IMAGE LOAD
            # -------------------------
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # contrast enhancement (çok önemli)
            img = apply_clahe(img)

            # -------------------------
            # 2. MASK PROCESS
            # -------------------------
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                # resize safe
                if mask.shape != img.shape:
                    mask = cv2.resize(
                        mask,
                        (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                # clean mask
                mask = clean_mask(mask)

                # expand ROI (context koruma)
                mask = expand_mask(mask, iterations=1)

                # final application
                masked = cv2.bitwise_and(img, img, mask=mask)

            else:
                masked = img

            # -------------------------
            # 3. SAVE (lossless)
            # -------------------------
            out_path = os.path.join(out_folder, name)
            cv2.imwrite(out_path, masked, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            total += 1

            if total % 1000 == 0:
                print(f"  -> {total} image processed")

    print(f"\nDONE ✔ Total images: {total}")
    print(f"Saved to: {output_dir}")

# RUN
process_dataset()