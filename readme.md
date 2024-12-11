# Sistem Rekomendasi Resep Masakan Indonesia Berbasis Machine Learning

Proyek ini bertujuan untuk mengembangkan sistem rekomendasi resep masakan Indonesia menggunakan teknik **machine learning**. Sistem ini memanfaatkan **TF-IDF**, **SVM (Support Vector Machine)**, dan **Random Forest** untuk memberikan rekomendasi resep berdasarkan bahan-bahan yang ada.

## Deskripsi
Sistem rekomendasi ini menggunakan beberapa algoritma pembelajaran mesin untuk mengklasifikasikan dan merekomendasikan resep masakan Indonesia. Dataset yang digunakan terdiri dari berbagai resep masakan dengan bahan-bahan dan langkah-langkah pembuatan yang telah disediakan. Sistem akan memberikan rekomendasi resep yang paling relevan berdasarkan input pengguna.

### Tujuan Proyek
- **Rekomendasi Resep**: Menyediakan rekomendasi resep berdasarkan input dari pengguna.
- **Pengklasifikasian Resep**: Menggunakan model klasifikasi untuk mengategorikan resep ke dalam kategori masakan tertentu.
- **Penggunaan TF-IDF dan Cosine Similarity**: Menghitung kemiripan antar resep berdasarkan bahan-bahan menggunakan teknik **TF-IDF** dan **Cosine Similarity**.

## Instalasi

Pastikan Anda memiliki Python 3 dan menginstal pustaka-pustaka berikut yang dibutuhkan untuk menjalankan proyek ini:

```bash
pip install -U scikit-learn
pip install plotly
pip install nltk
pip install ipywidgets
pip install xgboost
```
### Pustaka yang Digunakan:
- **scikit-learn**: Untuk pembuatan model klasifikasi dan evaluasi.
- **plotly**: Untuk visualisasi interaktif (meskipun tidak digunakan langsung dalam contoh ini).
- **nltk**: Untuk pengolahan teks, stopwords, dan pemrosesan bahasa alami.
- **ipywidgets**: Untuk membuat antarmuka pengguna interaktif di Google Colab.
- **xgboost**: Untuk algoritma XGBoost yang digunakan dalam klasifikasi (meskipun tidak digunakan dalam semua model di proyek ini).

## Proses

### 1. **Menggabungkan Dataset**
Dataset resep masakan Indonesia diunggah dalam beberapa file CSV, yang kemudian digabungkan menjadi satu DataFrame untuk analisis lebih lanjut.

### 2. **Preprocessing Data**
Proses preprocessing mencakup:
- Mengganti nilai **NaN** pada kolom **'Ingredients'** dan **'Steps'** dengan string kosong.
- Mengubah teks pada kolom **'Title'** menjadi huruf kecil.
- Membuat kolom baru yang menggabungkan **'Title'**, **'Ingredients'**, dan **'Steps'** menjadi satu kolom teks yang akan digunakan untuk membangun matriks **TF-IDF**.

### 3. **Membangun TF-IDF Matrix**
Menggunakan **TF-IDF Vectorizer** untuk mengonversi teks menjadi representasi numerik yang dapat diproses oleh model klasifikasi. Matrik **TF-IDF** ini digunakan untuk menghitung kemiripan antar resep menggunakan **Cosine Similarity**.

### 4. **Pelatihan Model**
Dua model klasifikasi digunakan dalam proyek ini:
- **SVM (Support Vector Machine)**: Untuk klasifikasi resep berdasarkan kategori.
- **Random Forest**: Sebagai model klasifikasi tambahan untuk membandingkan kinerjanya dengan SVM.

### 5. **Evaluasi Model**
Setelah melatih model, sistem menghitung **accuracy** dan memberikan evaluasi lengkap menggunakan **classification_report** dari **sklearn** yang mencakup metrik seperti **precision**, **recall**, dan **f1-score**.

### 6. **Rekomendasi Resep**
Fungsi **`recommend_recipe()`** menerima input berupa nama resep dan memberikan rekomendasi resep yang paling mirip berdasarkan bahan-bahan yang ada.

### 7. **Antarmuka Pengguna Interaktif**
Dengan menggunakan **ipywidgets**, pengguna dapat memasukkan nama resep yang mereka inginkan, dan sistem akan menampilkan 5 resep teratas yang relevan dengan resep yang diminta.

### 8. **Menyimpan Model**
Model yang telah dilatih, termasuk **TF-IDF Vectorizer**, **SVM model**, dan **Random Forest model**, disimpan menggunakan **pickle** untuk digunakan kembali di masa depan.
