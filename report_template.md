# LAPORAN TUGAS TRANSFER LEARNING

## 1. Judul
Klasifikasi Arah Gerak Robot Maqueen Micro:bit Menggunakan Transfer Learning ResNet-18

## 2. Latar Belakang
Robot Maqueen berbasis Micro:bit dapat digunakan untuk berbagai aplikasi robotika, termasuk navigasi dan obstacle avoidance. Pada tugas ini, computer vision digunakan untuk mengklasifikasikan kondisi navigasi dari citra kamera menjadi forward, left, right, dan stop.

## 3. Dataset
Dataset yang digunakan adalah End-to-end dataset dari Mendeley Data sebagai data awal/referensi. Dataset memiliki empat kelas yang sesuai dengan task navigasi robot. Data tambahan dari kamera robot sendiri dianjurkan untuk mengurangi perbedaan domain.

## 4. Metode
Model ResNet-18 pretrained ImageNet digunakan dengan tiga pendekatan:
- Feature extraction
- Fine-tuning parsial
- Training from scratch

Input diubah menjadi 224x224 RGB dan dinormalisasi menggunakan mean [0.485, 0.456, 0.406] dan std [0.229, 0.224, 0.225].

## 5. Hasil
Isi berdasarkan `results/summary.csv`.

| Mode | Best Val Accuracy | Waktu Training |
|---|---:|---:|
| Feature | ... | ... |
| Partial | ... | ... |
| Scratch | ... | ... |

Lampirkan grafik `results/accuracy_curve.png`.

## 6. Analisis
Bandingkan ketiga mode. Jelaskan apakah transfer learning memberikan keuntungan pada dataset kecil. Jelaskan juga kemungkinan overfitting, data leakage, dan domain shift.

## 7. Latensi
Isi berdasarkan output `latency.py`.

- Device: ...
- Latency: ... ms/frame
- FPS: ...

## 8. Kesimpulan
Transfer learning dengan ResNet-18 digunakan untuk membangun sistem klasifikasi navigasi robot Maqueen. Kesimpulan akhir harus dibuat berdasarkan hasil eksperimen aktual, bukan angka yang dibuat-buat.

## 9. Referensi
WAGA, A. (2024). End-to-end dataset. Mendeley Data. DOI: 10.17632/v2cb552j8m.1.

RET503 Pertemuan 3 - Transfer Learning dan Fine-Tuning.
