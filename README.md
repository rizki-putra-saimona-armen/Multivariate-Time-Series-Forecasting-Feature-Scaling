# Multivariate Time Series Forecasting & Feature Scaling

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Model-GRU-9B59B6?style=flat-square" />
  <img src="https://img.shields.io/badge/Task-Time%20Series%20Forecasting-2ECC71?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Completed-success?style=flat-square" />
</p>

Notebook ke-8 dari seri pembelajaran Deep Learning ini melanjutkan studi kasus *time series forecasting* dari model **univariate** ke **multivariate** — dengan menambahkan fitur musiman dari tanggal, membahas isu *data leakage* saat scaling, lalu membangun model **GRU** untuk memprediksi suhu minimum harian beberapa langkah ke depan.

---

##  Daftar Isi

- [Tentang Proyek](#-tentang-proyek)
- [Dataset](#-dataset)
- [Feature Engineering](#-feature-engineering)
- [Arsitektur Model](#-arsitektur-model)
- [Training & Hasil](#-training--hasil)
- [Forecasting Multi-Step](#-forecasting-multi-step)
- [Insight & Pembelajaran](#-insight--pembelajaran)
- [Cara Menjalankan](#-cara-menjalankan)
- [Struktur Proyek](#-struktur-proyek)
- [Tech Stack](#-tech-stack)

---

##  Tentang Proyek

Pada notebook-notebook sebelumnya, forecasting suhu hanya mengandalkan **1 fitur** yaitu nilai suhu itu sendiri (*univariate*). Di sini, kolom `Date` diolah lebih lanjut menjadi fitur baru yang merepresentasikan **musim (quarter)**, sehingga dataset berubah menjadi **multivariate** — dengan asumsi pola suhu berulang secara kuartalan sepanjang tahun.

Fokus utama notebook ini:

| Aspek | Deskripsi |
|---|---|
|  Transformasi fitur | Ekstraksi `quarter` dari `Date`, lalu one-hot encoding |
|  Scaling | Diskusi kritis soal risiko *data leakage* saat scaling manual |
|  Model | GRU (Gated Recurrent Unit) sebagai penerus RNN/LSTM di seri sebelumnya |
|  Forecasting | Eksperimen prediksi multi-step dengan berbagai panjang konteks (`n_prior`) & horizon (`n_forecast`) |

---

##  Dataset

Menggunakan dataset klasik **Daily Minimum Temperatures** (Melbourne, Australia):

- **Jumlah data:** 3.650 baris (10 tahun data harian, 1981–1990)
- **Kolom awal:** `Date`, `Temp`
- **Split:** 80% train (2.920 baris) · 20% test (730 baris), tanpa *shuffle* (menjaga urutan waktu)

---

##  Feature Engineering

1. **Ekstraksi kuartal dari tanggal**
   ```python
   df["quarter"] = df.Date.dt.quarter
   ```
2. **One-hot encoding** — kuartal diperlakukan sebagai data kategorik (lebih masuk akal secara semantik dibanding numerik):
   ```python
   df = pd.get_dummies(df, columns=["quarter"], dtype=float)
   ```
   Hasilnya, dataset yang semula 1 fitur (`Temp`) kini memiliki 5 fitur: `Temp`, `quarter_1`, `quarter_2`, `quarter_3`, `quarter_4`.

3. **Catatan soal scaling** — notebook ini secara sengaja membahas potensi *data leakage* ketika scaling manual dihitung dari mean/std seluruh dataset (bukan hanya train set). Untuk kasus ini scaling akhirnya dilewati, karena secara visual standar deviasi suhu relatif stabil sepanjang waktu — namun disebutkan eksplisit bahwa pada kasus umum, pendekatan yang lebih aman adalah **fit scaler hanya di data train** (misalnya via `sklearn.preprocessing`).

---

##  Arsitektur Model

Model menggunakan **GRU** (`nn.GRU`) dari PyTorch, diikuti fully-connected layer untuk memetakan output ke target prediksi:

```python
class GRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x, hidden = self.rnn(x, hidden)
        x = self.fc(x[:, -1, :])   # hanya timestep terakhir
        return x, hidden
```

**Konfigurasi:**

| Parameter | Nilai |
|---|---|
| `input_size` | 5 (Temp + 4 kolom quarter) |
| `hidden_size` | 128 |
| `num_layers` | 2 |
| `dropout` | 0.2 |
| `seq_len` | 14 |
| Loss function | MSELoss |
| Optimizer | AdamW (lr = 0.0005) |
| Batch size | 32 |
| Callback | `jcopdl.callback.Callback` (checkpoint, early stopping, live plotting) |

>  Catatan kecil: folder checkpoint model masih bernama `model/LSTM` — sisa penamaan dari iterasi sebelumnya, walau arsitektur final yang dipakai adalah **GRU**.

---

##  Training & Hasil

Model dilatih dengan skema *early stopping* berbasis `test_cost`. Ringkasan progresnya:

| Epoch | Train Cost | Test Cost | Keterangan |
|---|---|---|---|
| 1 | 42.86 | 20.03 | Awal training |
| 5 | 7.43 | 6.47 | Loss menurun cepat |
| 10 | 6.19 | 4.99 | Mulai stabil |
| 20 | 5.96 | 4.64 | Konvergen |
| **25** | **5.84** | **4.55** | **Best checkpoint** |
| 30 | 5.84 | 4.56 | Early stop terpicu |

Training berhenti otomatis di **epoch 30** dengan **best test cost ≈ 4.5504**, dan model terbaik tersimpan otomatis lewat callback.

---

##  Forecasting Multi-Step

Setelah training, model diuji untuk memprediksi banyak langkah ke depan sekaligus menggunakan fungsi `pred4pred`, dengan dua skenario:

| Skenario | Konteks (`n_prior`) | Horizon (`n_forecast`) | Observasi |
|---|---|---|---|
| 1 | 10 data awal | 140 langkah | Prediksi mulai melenceng signifikan di horizon jauh |
| 2 | 30 data awal | 120 langkah | Konteks lebih panjang → prediksi awal lebih stabil, namun tetap melenceng di ujung horizon |

---

##  Insight & Pembelajaran

Beberapa catatan reflektif dari eksplorasi notebook ini:

- **Feature engineering > menambah kompleksitas model.** Menambahkan fitur `quarter` yang bermakna terbukti lebih membantu daripada sekadar memperbesar arsitektur — inti dari machine learning adalah mencari pola, dan tugas kita adalah menyediakan informasi yang relevan.
- **Ketidakpastian meningkat seiring horizon prediksi.** Semakin jauh titik waktu yang diprediksi dari titik referensi terakhir, semakin besar potensi *drift* / error-nya — ini adalah karakteristik alami forecasting, bukan tanda model yang gagal.
- **Overfitting di time series forecasting adalah hal yang lumrah diantisipasi.** Ekspektasi terhadap prediksi jangka panjang perlu realistis; akurasi sempurna di semua horizon nyaris mustahil dicapai.

---

##  Cara Menjalankan

```bash
# 1. Clone repository
git clone <repo-url>
cd part-8-multivariate-scaling

# 2. Install dependencies
pip install torch pandas numpy matplotlib scikit-learn tqdm
pip install jcopdl jcopml

# 3. Jalankan notebook
jupyter notebook "Part_8_-_Multivariate_and_scaling.ipynb"
```

>  Path dataset (`daily_min_temp.csv`) di notebook masih menggunakan path lokal Windows — sesuaikan ke path relatif sebelum menjalankan ulang di environment lain.

---

##  Struktur Proyek

```
 part-8-multivariate-scaling
 ┣  Multivariate_and_scaling.ipynb
 ┣  data
 ┃  daily_min_temp.csv
 ┣  model
 ┃ ┗  LSTM/            # checkpoint model GRU tersimpan di sini
 ┗  README.md
```

---

##  Tech Stack

| Kategori | Tools |
|---|---|
| Bahasa | Python 3.12 |
| Deep Learning | PyTorch, `jcopdl` |
| Data Processing | Pandas, NumPy |
| Visualisasi | Matplotlib |
| Preprocessing | scikit-learn (`train_test_split`) |
| Training Utility | tqdm, jcopdl Callback (early stopping & checkpointing) |

---

