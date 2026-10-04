# Draf Dokumen Desain Pj1

## Misi proyek
Membangun sistem persepsi berbasis kamera untuk membantu robot Maqueen Micro:bit menentukan arah gerak berdasarkan citra lingkungan. Model klasifikasi akan menghasilkan empat kelas keputusan: forward, left, right, dan stop.

## Kelas objek/keputusan
1. forward
2. left
3. right
4. stop

## Kamera
Kamera RGB yang menghadap arah gerak robot. Resolusi saat deployment harus konsisten dengan preprocessing saat training.

## Unit komputasi
Training dilakukan pada PC/laptop menggunakan PyTorch. Micro:bit digunakan sebagai pengendali robot Maqueen.

## Target kinerja
- Validation accuracy: dicatat dari hasil aktual eksperimen.
- Inference latency: diukur dengan `latency.py`.
- Target awal: mendekati kebutuhan real-time; angka final ditentukan dari pengukuran perangkat.

## Kandidat model
1. ResNet-18 — dipilih sebagai model utama karena menjadi model praktikum dan relatif ringan.
2. MobileNetV3-Small — kandidat pembanding untuk deployment dengan komputasi lebih terbatas.

## Strategi transfer learning
Eksperimen:
- feature extraction
- fine-tuning parsial
- scratch

Hipotesis: feature extraction dan fine-tuning parsial akan lebih baik daripada scratch pada dataset yang relatif kecil.

## Rencana data
Minimal 50 citra per kelas. Data harus mencakup variasi jarak, posisi, cahaya, latar, dan kondisi sulit.

## Risiko dan mitigasi
1. Data leakage -> pisahkan data berdasarkan sesi.
2. Overfitting -> augmentasi dan transfer learning.
3. Latensi tinggi -> ukur latency dan pertimbangkan model lebih ringan.
4. Domain shift antara dataset publik dan kamera Maqueen -> tambahkan data dari kamera/lingkungan robot sendiri.
