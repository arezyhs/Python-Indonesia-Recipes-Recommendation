# Indonesian Recipe Finder

Machine Learning system untuk rekomendasi resep masakan Indonesia menggunakan TF-IDF dan SVM.

## Overview

Sistem rekomendasi resep yang menggunakan Natural Language Processing dan Machine Learning untuk membantu menemukan resep masakan Indonesia berdasarkan input pengguna.

**Features:**
- Search resep dengan keyword
- Klasifikasi kategori makanan (SVM 88.3% accuracy) 
- Web interface dengan Streamlit
- Dataset 15,641 resep Indonesia

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run aplikasi:
```bash
streamlit run app/streamlit_app.py
```

3. Buka browser: `http://localhost:8501`

## Tech Stack

- **ML**: scikit-learn, TF-IDF, SVM, Random Forest
- **Web**: Streamlit, Plotly
- **Data**: Pandas, NLTK

## Dataset

| Kategori | Jumlah |
|----------|--------|
| Tahu | 2,419 |
| Ayam | 2,092 |
| Tempe | 1,777 |
| Udang | 1,732 |
| Sapi | 1,593 |
| Kambing | 1,458 |
| Telur | 1,219 |
| Ikan | 856 |

**Total: 15,641 resep**

## Author

Portfolio project untuk showcase kemampuan Machine Learning dan NLP dalam domain kuliner Indonesia.
