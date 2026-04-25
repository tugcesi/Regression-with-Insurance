# 🛡️ Regression with an Insurance Dataset

Bu proje, sigorta müşterilerinin **Premium Amount (Prim Tutarı)** değerini tahmin eden bir makine öğrenmesi uygulamasıdır.

---

## 📂 Proje Yapısı

```
├── regression-with-an-insurance-dataset.ipynb  # Analiz & model geliştirme notebook'u
├── save_model.py                                # Model eğitimi ve artifact kaydetme
├── app.py                                       # Streamlit web uygulaması
├── train.csv                                    # Eğitim verisi (Kaggle'dan indir)
├── requirements.txt                             # Bağımlılıklar
└── README.md
```

---

## 📊 Veri Seti

Veri seti [Kaggle](https://www.kaggle.com/competitions/sch2-reg-2026-d3-3/data) üzerinden erişilebilir.

| Özellik | Açıklama |
|---|---|
| Age | Müşteri yaşı |
| Gender | Cinsiyet |
| Annual Income | Yıllık gelir |
| Marital Status | Medeni durum |
| Number of Dependents | Bakmakla yükümlü kişi sayısı |
| Education Level | Eğitim seviyesi |
| Occupation | Meslek |
| Health Score | Sağlık skoru |
| Location | Konum (Urban/Suburban/Rural) |
| Policy Type | Sigorta poliçe tipi |
| Previous Claims | Geçmiş hasar sayısı |
| Vehicle Age | Araç yaşı |
| Credit Score | Kredi skoru |
| Insurance Duration | Sigorta süresi (yıl) |
| Policy Start Date | Poliçe başlangıç tarihi |
| Customer Feedback | Müşteri geri bildirimi |
| Smoking Status | Sigara kullanımı |
| Exercise Frequency | Egzersiz sıklığı |
| Property Type | Mülk tipi |
| **Premium Amount** | 🎯 **Hedef değişken** |

- **Train:** 1.200.000 satır
- **Test:** 800.000 satır

---

## ⚙️ Feature Engineering

Notebook'ta uygulanan özellik mühendisliği adımları:

- `Log_Annual_Income` → Yıllık gelirin log dönüşümü
- `Claims_Per_Year` → Yıllık başına hasar oranı
- `Age_Health_Interaction` → Yaş × Sağlık skoru (risk proxy)
- `Income_Credit_Ratio` → Gelir / Kredi skoru oranı
- `Total_Risk_Duration` → Araç yaşı + Sigorta süresi
- `Previous_Claims_Missing` → Eksik değer flag'i

**Encoding:**
- Binary: `Gender`, `Smoking Status`
- Ordinal: `Education Level`, `Customer Feedback`, `Exercise Frequency`
- One-Hot: `Marital Status`, `Occupation`, `Location`, `Policy Type`, `Property Type`

**Target dönüşümü:** `log1p(Premium Amount)` → tahmin sonrası `np.expm1()` ile geri alınır.

---

## 🤖 Model

| Model | R² | RMSE | MAE |
|---|---|---|---|
| XGBoost | En iyi | - | - |
| GradientBoosting | - | - | - |
| DecisionTree | - | - | - |
| LinearRegression | - | - | - |

**Final model:** `XGBRegressor(n_jobs=-1, random_state=42)`

---

## 🚀 Kurulum & Çalıştırma

### 1. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

### 2. Veri Setini İndir

`train.csv` dosyasını [Kaggle](https://www.kaggle.com/competitions/sch2-reg-2026-d3-3/data)'dan indirip proje dizinine koy.

### 3. Modeli Eğit

```bash
python save_model.py
```

Bu adım şu dosyaları oluşturur:
- `model.joblib`
- `feature_columns.joblib`
- `num_medians.joblib`
- `cat_modes.joblib`
- `ohe_categories.joblib`

### 4. Uygulamayı Başlat

```bash
streamlit run app.py
```

Tarayıcında `http://localhost:8501` adresine git.

---

## 🖥️ Uygulama Özellikleri

- 👤 Kişisel bilgi girişi (yaş, cinsiyet, eğitim, meslek...)
- 🏥 Sağlık & yaşam tarzı bilgileri
- 💰 Finansal bilgiler (gelir, kredi skoru...)
- 🚗 Araç & sigorta detayları
- 📊 Tahmin sonucu + Plotly gauge chart
- 📈 Feature importance görselleştirmesi

---

## 📦 Bağımlılıklar

```
streamlit
pandas
numpy
xgboost
scikit-learn
joblib
plotly
```

---

## 📌 Kaynak

- Kaggle Competition: [sch2-reg-2026-d3-3](https://www.kaggle.com/competitions/sch2-reg-2026-d3-3)
