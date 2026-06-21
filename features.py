"""
Pipeline ekstraksi fitur — HARUS identik dengan notebook training
(Klasifikasi_Kulit_Wajah_SVM__acc_.ipynb) agar model SVM yang sudah
dilatih dapat dipakai dengan benar saat inferensi.

Fitur Warna (36): RGB mean/std (6) + RGB histogram 8-bin x3 channel (24)
                  + HSV mean/std (6)
Fitur Tekstur (14): GLCM 4 arah -> contrast, correlation, energy,
                    homogeneity (4) + LBP histogram 10-bin (10)
Total: 50 dimensi
"""

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

IMG_SIZE = (224, 224)


def preprocess_image_array(img_bgr):
    """Terima gambar BGR (numpy array, dari cv2.imread atau konversi PIL->cv2),
    resize, dan konversi ke ruang warna RGB, HSV, Grayscale."""
    if img_bgr is None:
        return None, None, None
    img_resized = cv2.resize(img_bgr, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    return img_rgb, img_hsv, img_gray


def extract_color_features(img_rgb, img_hsv):
    """
    Ekstraksi fitur warna dari ruang RGB dan HSV.
    Output: 36 fitur (6 mean/std RGB + 24 histogram RGB + 6 mean/std HSV)
    """
    features = []
    # RGB: Mean & Std per channel
    for ch in range(3):
        features.append(np.mean(img_rgb[:, :, ch]) / 255.0)
        features.append(np.std(img_rgb[:, :, ch]) / 255.0)
    # RGB: Histogram 8 bin per channel
    for ch in range(3):
        hist = np.histogram(img_rgb[:, :, ch], bins=8, range=(0, 255))[0]
        features.extend(hist / float(img_rgb[:, :, ch].size))
    # HSV: Mean & Std per channel
    hsv_scale = [179.0, 255.0, 255.0]
    for ch in range(3):
        features.append(np.mean(img_hsv[:, :, ch]) / hsv_scale[ch])
        features.append(np.std(img_hsv[:, :, ch]) / hsv_scale[ch])
    return features  # total: 36 fitur


def extract_texture_features(img_gray):
    """
    Ekstraksi fitur tekstur menggunakan GLCM (4 arah) dan LBP.
    Output: 14 fitur (4 GLCM + 10 LBP histogram)
    """
    features = []
    # GLCM 4 arah: 0°, 45°, 90°, 135°
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(
        img_gray, distances=[1], angles=angles,
        levels=256, symmetric=True, normed=True
    )
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
        val = graycoprops(glcm, prop).mean()
        if prop == 'contrast':
            val = val / 1000.0
        features.append(val)
    # LBP histogram
    lbp = local_binary_pattern(img_gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    features.extend(lbp_hist)
    return features  # total: 14 fitur


def extract_features_pipeline(img_bgr):
    """
    Pipeline lengkap: preprocessing -> fitur warna + tekstur -> fusion (50 dimensi).

    Parameter:
        img_bgr: numpy array gambar dalam format BGR (hasil cv2.imread,
                 atau hasil konversi dari PIL via cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    Return:
        list berisi 50 nilai fitur, atau None jika gambar tidak valid.
    """
    img_rgb, img_hsv, img_gray = preprocess_image_array(img_bgr)
    if img_rgb is None:
        return None
    return extract_color_features(img_rgb, img_hsv) + extract_texture_features(img_gray)


def extract_features_from_path(image_path):
    """Versi pipeline yang menerima path file (sama seperti di notebook)."""
    img_bgr = cv2.imread(image_path)
    return extract_features_pipeline(img_bgr)
