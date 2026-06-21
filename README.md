# Klasifikasi Permasalahan Kulit Wajah Menggunakan Ekstraksi Fitur Warna dan Tekstur Berbasis Machine Learning dengan Algoritma Support Vector Machine (SVM)

## Identitas Proyek

| Keterangan | Isi |
|---|---|
| Mata Kuliah | Pembelajaran Mesin |
| Topik | Klasifikasi Citra Medis / Dermatologi Komputasional |
| Algoritma | Support Vector Machine (SVM) |
| Anggota Kelompok | Putu Eka Febriani (E1E124013), Farid Khandra (E1E124006) |
| Dosen Pembimbing | Rizal Adi Saputra, S.T., M.T |

---

## 1. Latar Belakang

Identifikasi permasalahan kulit wajah di kalangan kecantikan masih banyak
dilakukan secara manual oleh tenaga ahli, sehingga membutuhkan waktu dan
sangat bergantung pada pengalaman profesional tenaga ahli. Perkembangan
teknologi *Machine Learning* dalam pengolahan citra digital memungkinkan
klasifikasi kondisi kulit wajah dilakukan secara otomatis menggunakan
fitur warna dan tekstur, sehingga menghasilkan sistem yang lebih cepat,
objektif, dan efisien.

## 2. Rumusan Masalah

1. Proses mengidentifikasi kulit wajah secara manual memerlukan waktu
   yang cukup lama.
2. Hasil identifikasi sangat bergantung pada pengalaman tenaga ahli,
   sehingga rawan subjektivitas.
3. Belum banyak penelitian yang menggabungkan fitur warna dan tekstur
   secara bersamaan untuk klasifikasi kulit wajah.
4. Diperlukan sistem klasifikasi yang otomatis, cepat, efisien, dan
   objektif.

## 3. Tujuan Penelitian

Mengembangkan sistem klasifikasi kondisi kulit wajah menggunakan fitur
warna dan tekstur berbasis *machine learning*, serta mengimplementasikan
algoritma **Support Vector Machine (SVM)** untuk menghasilkan model
klasifikasi yang akurat dan dapat diakses melalui aplikasi web.

## 4. Manfaat Penelitian

- Membantu mengidentifikasi kondisi awal kulit wajah secara cepat.
- Menghemat waktu dan meningkatkan efisiensi layanan di klinik
  kecantikan.
- Memberikan hasil yang lebih objektif dan konsisten dibanding penilaian
  manual.
- Dapat menjadi sistem pendukung keputusan bagi tenaga medis/kecantikan.

---

## 5. Dataset

Dataset bersumber dari **Kaggle**, terdiri atas citra kulit wajah yang
dikelompokkan ke dalam 6 kelas permasalahan kulit:

| No | Kelas | Deskripsi |
|----|-------|-----------|
| 1 | `acne` | Jerawat inflamasi |
| 2 | `dark spots` | Bintik hitam / hiperpigmentasi |
| 3 | `Redness` | Kemerahan pada kulit |
| 4 | `pores` | Pori-pori membesar |
| 5 | `wrinkles` | Kerutan |
| 6 | `normal` | Kulit normal (tidak ada masalah) |

Data dibagi menjadi data *training* (untuk pelatihan model) dan data
*testing* (untuk evaluasi performa model), dengan proporsi 80:20.

## 6. Metodologi Penelitian

Penelitian ini menerapkan alur metodologi sebagai berikut:

```
Input Citra Kulit Wajah
    ↓
Preprocessing (Resize 224×224 + Konversi Ruang Warna)
    ↓
Ekstraksi Fitur Warna  : RGB (Mean, Std, Histogram) + HSV (Mean, Std)
    ↓
Ekstraksi Fitur Tekstur: GLCM (4 arah) + LBP Histogram
    ↓
Feature Fusion (50 dimensi)
    ↓
Normalisasi (StandardScaler)
    ↓
Hyperparameter Tuning (GridSearchCV, 5-Fold Cross Validation)
    ↓
Klasifikasi SVM (kernel terbaik)
    ↓
Evaluasi & Visualisasi Hasil
    ↓
Implementasi pada Aplikasi Web (Streamlit)
```

### 6.1 Preprocessing

Setiap citra masukan diubah ukurannya menjadi 224×224 piksel, kemudian
dikonversi ke tiga ruang warna: RGB, HSV, dan *grayscale*, sebagai dasar
ekstraksi fitur pada tahap berikutnya.

### 6.2 Ekstraksi Fitur Warna (36 dimensi)

- **RGB**: nilai rata-rata (*mean*) dan simpangan baku (*std*) tiap
  kanal warna (6 fitur), serta histogram 8-bin tiap kanal (24 fitur).
- **HSV**: nilai rata-rata dan simpangan baku tiap kanal (6 fitur).

### 6.3 Ekstraksi Fitur Tekstur (14 dimensi)

- **GLCM (*Gray-Level Co-occurrence Matrix*)** pada 4 arah sudut (0°,
  45°, 90°, 135°), menghasilkan 4 properti tekstur: *contrast*,
  *correlation*, *energy*, dan *homogeneity*.
- **LBP (*Local Binary Pattern*)**: histogram pola tekstur lokal dengan
  10 bin.

Kedua kelompok fitur digabungkan (*feature fusion*) menjadi satu vektor
berdimensi **50**, kemudian dinormalisasi menggunakan `StandardScaler`
sebelum masuk ke tahap klasifikasi.

### 6.4 Pelatihan Model (SVM)

Model klasifikasi dilatih menggunakan algoritma **Support Vector Machine
(SVM)**. Pencarian kombinasi *hyperparameter* terbaik (kernel, nilai C,
dan gamma) dilakukan dengan **GridSearchCV** menggunakan skema
**5-Fold Stratified Cross Validation**, sehingga parameter yang dipilih
benar-benar teroptimasi secara otomatis, bukan ditentukan secara manual.

### 6.5 Hasil Model Terbaik

| Parameter | Nilai |
|---|---|
| Kernel | RBF |
| C | 10 |
| Gamma | scale |

### 6.6 Hasil Evaluasi Model

Evaluasi dilakukan pada data uji (*testing set*) yang terpisah dari data
latih, menggunakan metrik standar klasifikasi:

| Metrik Evaluasi | Nilai |
|---|---|
| Akurasi | 84.12% |
| Presisi (*Macro*) | 84.46% |
| Recall (*Macro*) | 84.17% |
| F1-Score (*Macro*) | 84.20% |

---

## 7. Implementasi — Aplikasi Web

Sebagai bentuk implementasi dari model yang telah dilatih, dibangun
sebuah aplikasi web interaktif menggunakan **Streamlit**, yang
memungkinkan pengguna melakukan klasifikasi kondisi kulit wajah secara
langsung melalui dua cara input:

1. **Upload Foto** — pengguna mengunggah gambar kulit wajah dari
   perangkat.
2. **Scan Langsung (Kamera)** — pengguna mengambil foto secara *real-time*
   melalui kamera perangkat (laptop/HP).

Aplikasi menampilkan hasil klasifikasi berupa kelas prediksi beserta
tingkat keyakinan (*confidence score*) untuk masing-masing dari 6 kelas,
sehingga hasil klasifikasi dapat dipertanggungjawabkan secara kuantitatif.

Desain antarmuka aplikasi disesuaikan dengan identitas visual poster
penelitian (warna maroon, *pink*, dan *cream*) agar konsisten dengan
luaran akademik lain dari penelitian ini.

### 7.1 Struktur Berkas Aplikasi

```
skinapp/
├── app.py                  # Aplikasi utama (antarmuka & logika prediksi)
├── features.py              # Pipeline ekstraksi fitur (identik dengan notebook pelatihan)
├── requirements.txt          # Daftar pustaka (library) yang digunakan
├── .streamlit/
│   └── config.toml            # Konfigurasi tema tampilan
└── saved_model/
    ├── model_svm.pkl            # Model SVM hasil pelatihan
    ├── scaler.pkl                 # Objek normalisasi (StandardScaler)
    ├── classes.pkl                  # Daftar label kelas
    ├── best_params.pkl                # Parameter terbaik hasil GridSearchCV
    └── metrics.pkl                       # Ringkasan metrik evaluasi model
```

> **Catatan teknis:** Pipeline ekstraksi fitur pada `features.py` dibuat
> identik dengan proses pada notebook pelatihan model (ukuran gambar,
> jumlah bin histogram, dan urutan fitur), agar model yang telah dilatih
> dapat digunakan secara konsisten saat proses inferensi pada aplikasi.

### 7.2 Tautan Aplikasi

Aplikasi telah dideploy secara publik melalui Streamlit Community Cloud
dan dapat diakses melalui tautan berikut:

> *(isi tautan aplikasi Streamlit kamu di sini setelah deploy selesai,
> contoh: https://skin-classification-app-xxxxx.streamlit.app)*

---

## 8. Batasan dan Catatan

Sistem ini dikembangkan untuk **keperluan akademik dan penelitian**
sebagai bentuk penerapan metode *machine learning* dalam klasifikasi
citra kulit wajah. Hasil klasifikasi yang ditampilkan **bukan merupakan
diagnosis medis** dan tidak dimaksudkan untuk menggantikan pemeriksaan
oleh dokter kulit (dermatolog) maupun tenaga medis profesional lainnya.