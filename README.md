# 🎓 Student Placement & Salary Prediction

Sistem machine learning end-to-end untuk memprediksi **apakah seorang mahasiswa akan mendapatkan pekerjaan (Placed / Not Placed)** dan **berapa estimasi gajinya**, berdasarkan profil akademik, skill, pengalaman, dan gaya hidup. Model di-deploy sebagai **REST API (FastAPI)** dan **aplikasi web (Streamlit)**, dengan eksperimen yang dilacak menggunakan **MLflow**.

🔗 **Live Demo:** [uts-model-deployment-app.streamlit.app](https://uts-model-deployment-app.streamlit.app/)

---

## 📌 Latar Belakang

Career Center kampus perlu mengetahui mahasiswa mana yang berisiko belum mendapat pekerjaan **sebelum mereka lulus**, agar bisa diberi intervensi lebih awal (pelatihan skill, rekomendasi magang, atau mentoring). Project ini membangun sistem *early warning* yang:

1. **Klasifikasi** – memprediksi peluang mahasiswa mendapat pekerjaan.
2. **Regresi** – mengestimasi gaji (dalam LPA / *Lakh Per Annum*) untuk mahasiswa yang diprediksi Placed.
3. **Rekomendasi** – memberikan saran singkat berdasarkan hasil prediksi.

> Dataset berasal dari studi kasus mata kuliah (Model Deployment). Pipeline dirancang agar dapat digunakan ulang dengan data *tracer study* kampus yang sebenarnya.

---

## 🗂️ Struktur Project

```
├── A.csv                    # Fitur profil mahasiswa
├── A_targets.csv            # Target: placement_status & salary_lpa
├── EDA_Modeling.ipynb       # Eksplorasi data & eksperimen model
├── pipeline.py              # Training pipeline + MLflow tracking
├── api_fastapi.py           # REST API (FastAPI)
├── frontend_streamlit.py    # UI Streamlit yang memanggil FastAPI
├── app_streamlit.py         # UI Streamlit standalone (versi yang di-deploy)
├── models/
│   ├── best_classifier.pkl
│   ├── best_regressor.pkl
│   └── feature_metadata.json
├── mlruns/                  # Log eksperimen MLflow
└── requirements.txt
```

---

## 📊 Dataset

| File | Isi |
|------|-----|
| `A.csv` | 22 fitur: gender, jurusan, CGPA, nilai kelas 10 & 12, backlog, jam belajar, kehadiran, jumlah proyek/magang/sertifikasi/hackathon, rating skill coding/komunikasi/aptitude, jam tidur, tingkat stres, kerja paruh waktu, pendapatan keluarga, tier kota, akses internet, ekstrakurikuler |
| `A_targets.csv` | `placement_status` (Placed / Not Placed) dan `salary_lpa` |

Kedua file digabung berdasarkan `Student_ID`.

---

## ⚙️ Alur Machine Learning

### 1. Feature Engineering
| Fitur Baru | Deskripsi |
|------------|-----------|
| `skill_composite` | Rata-rata skill coding, komunikasi, dan aptitude |
| `academic_score` | Kombinasi berbobot CGPA, kehadiran, dan jam belajar |
| `experience_score` | Magang ×2 + proyek + sertifikasi + hackathon |
| `healthy_sleep` | 1 jika jam tidur antara 6–9 jam |
| `high_achiever` | 1 jika CGPA ≥ 8.0 dan skill coding ≥ 7 |

### 2. Preprocessing
- **Numerik:** imputasi median + `StandardScaler`
- **Kategorikal:** imputasi modus + `OneHotEncoder`
- Semua digabung dalam `ColumnTransformer` di dalam `sklearn.Pipeline`, sehingga preprocessing ikut tersimpan bersama model.

### 3. Modeling
- **Klasifikasi:** Logistic Regression, Random Forest, Gradient Boosting (split 80/20 dengan `stratify`, `class_weight='balanced'` untuk menangani ketidakseimbangan kelas). Model terbaik dipilih berdasarkan **F1-Score**.
- **Regresi:** Ridge Regression, Random Forest, Gradient Boosting. Dilatih **hanya pada mahasiswa yang Placed**, karena hanya mereka yang memiliki gaji nyata. Model terbaik dipilih berdasarkan **R²**.

### 4. Experiment Tracking
Setiap model dicatat ke **MLflow** (parameter, metrik, dan artefak model) dalam dua eksperimen: `Student_Placement_Classification` dan `Student_Salary_Regression`.

---

## 📈 Hasil

### Klasifikasi (Placement)
| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 0.8250 | 0.8460 | **0.9108** |
| Random Forest | **0.8850** | 0.8616 | 0.8950 |
| **Gradient Boosting** ✅ | 0.8820 | **0.8739** | 0.9060 |

### Regresi (Salary)
| Model | RMSE | R² |
|-------|------|----|
| Ridge Regression | 1.4069 | 0.7717 |
| Random Forest | 1.4828 | 0.7464 |
| **Gradient Boosting** ✅ | **1.3993** | **0.7741** |

**Gradient Boosting** terpilih untuk kedua tugas. Model regresi mampu menjelaskan sekitar **77% variasi gaji** dengan rata-rata error sekitar **1.4 LPA**.

---

## 🚀 Cara Menjalankan

### 1. Instalasi
```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### 2. Training Ulang Model (opsional)
```bash
python pipeline.py
```
Model terbaik akan tersimpan di folder `models/`. Untuk melihat hasil eksperimen:
```bash
mlflow ui
```
Lalu buka `http://127.0.0.1:5000`.

### 3. Menjalankan Aplikasi Streamlit (standalone)
```bash
streamlit run app_streamlit.py
```

### 4. Menjalankan FastAPI + Frontend
```bash
# Terminal 1
uvicorn api_fastapi:app --host 127.0.0.1 --port 8000

# Terminal 2
streamlit run frontend_streamlit.py
```
Dokumentasi API interaktif tersedia di `http://127.0.0.1:8000/docs`.

---

## 🔌 API Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/` | Info API |
| GET | `/health` | Status model yang dimuat |
| POST | `/predict/placement` | Prediksi Placed / Not Placed + probabilitas & tingkat keyakinan |
| POST | `/predict/salary` | Estimasi gaji (LPA) beserta rentang estimasi |
| POST | `/predict/both` | Prediksi placement + gaji + rekomendasi |

**Contoh request:**
```json
{
  "gender": "Male",
  "branch": "Computer Science",
  "cgpa": 8.5,
  "tenth_percentage": 85.0,
  "twelfth_percentage": 82.0,
  "backlogs": 0,
  "study_hours_per_day": 6.0,
  "attendance_percentage": 90.0,
  "projects_completed": 5,
  "internships_completed": 2,
  "coding_skill_rating": 8,
  "communication_skill_rating": 7,
  "aptitude_skill_rating": 8,
  "hackathons_participated": 3,
  "certifications_count": 4,
  "sleep_hours": 7.0,
  "stress_level": 4,
  "part_time_job": "No",
  "family_income_level": "Medium",
  "city_tier": "Tier 1",
  "internet_access": "Yes",
  "extracurricular_involvement": "High"
}
```

---

## 🛠️ Tech Stack

- **Bahasa:** Python
- **Data & ML:** pandas, NumPy, scikit-learn
- **Visualisasi:** Matplotlib, Seaborn
- **Experiment Tracking:** MLflow
- **Deployment:** FastAPI, Uvicorn, Streamlit, Streamlit Community Cloud

---

## 🔮 Pengembangan Selanjutnya

- Adaptasi ke konteks Indonesia (IPK skala 4, gaji dalam Rupiah, data *tracer study*).
- Hyperparameter tuning dan model boosting lain (XGBoost / LightGBM).
- Penjelasan prediksi per mahasiswa dengan SHAP.
- Containerization dengan Docker.
