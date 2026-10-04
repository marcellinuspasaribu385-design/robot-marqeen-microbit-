# Sumber Data

## Dataset utama/referensi
**End-to-end dataset — Mendeley Data**

- URL: https://data.mendeley.com/datasets/v2cb552j8m/1
- DOI: 10.17632/v2cb552j8m.1
- Contributor: Abderrahim WAGA
- Tahun: 2024
- Kelas: forward, left, right, stop
- Domain: mobile robot navigation / obstacle avoidance / autonomous navigation
- Lisensi: CC BY 4.0

Dataset ini digunakan sebagai data awal karena task-nya langsung berkaitan dengan navigasi robot menggunakan kamera tunggal.

## Data robot sendiri
Untuk hasil yang paling sesuai dengan Maqueen, tambahkan citra yang diambil dari kamera pada robot/lingkungan operasi sebenarnya. Simpan minimal 50 citra per kelas dan variasikan:
- jarak dekat/sedang/jauh
- posisi objek di tengah/tepi
- orientasi robot
- terang/redup/bayangan
- latar
- kondisi sulit seperti occlusion dan motion blur

Simpan metadata pengambilan agar train/validation dapat dipisahkan berdasarkan sesi.
