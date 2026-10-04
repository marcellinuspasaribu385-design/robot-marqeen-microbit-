# Transfer Learning ResNet-18 untuk Robot Maqueen Micro:bit

## 1. Judul
**Klasifikasi Arah Gerak Robot Maqueen Micro:bit Menggunakan Transfer Learning ResNet-18**

## 2. Tujuan
Membangun model computer vision yang mengklasifikasikan kondisi navigasi robot menjadi:
- `forward`
- `left`
- `right`
- `stop`

Model utama yang digunakan adalah **ResNet-18 pretrained ImageNet**. Eksperimen membandingkan:
1. Feature extraction
2. Fine-tuning parsial
3. Training from scratch

Struktur dan parameter eksperimen mengikuti materi RET503 Pertemuan 3.

## 3. Dataset
Dataset awal yang digunakan adalah **End-to-end dataset** dari Mendeley Data:
https://data.mendeley.com/datasets/v2cb552j8m/1

Dataset tersebut dibuat untuk navigasi mobile robot menggunakan satu kamera dan memiliki empat kelas: forward, left, right, dan stop.

Untuk memenuhi konteks robot Maqueen, dataset publik sebaiknya digunakan sebagai **data awal/referensi**, lalu tambahkan data yang diambil dari kamera yang akan dipasang pada robot. Target minimal praktikum: **>= 50 citra per kelas**.

> Jangan mengarang angka akurasi. Jalankan `train.py` dan masukkan hasil aktual ke laporan.

## 4. Arsitektur
- Backbone: ResNet-18
- Bobot pretrained: ImageNet
- Input: 224 x 224 RGB
- Normalisasi:
  - mean = [0.485, 0.456, 0.406]
  - std = [0.229, 0.224, 0.225]

## 5. Eksperimen
| Mode | Bobot awal | Layer dilatih | Learning rate |
|---|---|---|---|
| feature | ImageNet | fc | 1e-3 |
| partial | ImageNet | layer4 + fc | layer4=1e-4, fc=1e-3 |
| scratch | acak | semua | 1e-3 |

10 epoch, augmentasi RandomResizedCrop, HorizontalFlip, ColorJitter, dan CosineAnnealingLR.

## 6. Struktur dataset
```text
data/
└── dataset_raw/
    ├── forward/
    ├── left/
    ├── right/
    └── stop/
```

Gunakan nama file yang menyimpan informasi sesi, misalnya:
`session01_terang_forward_001.jpg`

Hal ini membantu mengurangi data leakage antara train dan validation.

## 7. Cara menjalankan
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate

pip install -r requirements.txt

python split.py
python train.py
python latency.py
```

Hasil training akan tersimpan di:
```text
results/
├── metrics.csv
├── accuracy_curve.png
└── best_model_<mode>.pth
```

## 8. Implementasi ke Maqueen
Model klasifikasi berjalan pada komputer/edge device yang terhubung dengan kamera. Hasil klasifikasi dapat dipetakan:
- forward -> robot maju
- left -> robot belok kiri
- right -> robot belok kanan
- stop -> robot berhenti

Micro:bit/Maqueen menangani kontrol motor sesuai perintah yang dikirim dari sistem vision.

Maqueen mendukung kontrol motor dan pembacaan sensor melalui library DFRobot Maqueen:
https://makecode.microbit.org/pkg/DFRobot/pxt-maqueen

## 9. Catatan penting
Materi RET503 menekankan bahwa frame berurutan yang sangat mirip tidak boleh dibagi sembarang ke train dan validation karena dapat menyebabkan data leakage. Pisahkan berdasarkan sesi/kondisi pengambilan.

## 10. Referensi
- RET503 Pertemuan 3 – Transfer Learning dan Fine-Tuning
- WAGA, A. (2024). End-to-end dataset. Mendeley Data. DOI: 10.17632/v2cb552j8m.1
- He et al. (2016). Deep Residual Learning for Image Recognition.
- DFRobot Maqueen documentation.
