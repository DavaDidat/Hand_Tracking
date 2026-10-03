# RETROLENS

App webcam **hand-tracking** berbasis Python: tangan lo jadi "portal" yang nampilin efek filter, plus mode foto dengan hasil polaroid di panel samping.

Dibangun dengan **MediaPipe Hand Landmarker** + **OpenCV**, single-file (`retrolens.py`, ~1050 baris).

## Fitur

- **Mode FILTER** — portal di sekitar tangan (hull 2D atau kubus 3D mengambang) yang hanya mem-filter bagian di dalamnya, bukan seluruh frame.
  - 11 filter: `dual-tone`, `thermal`, `sketch`, `pixelate`, `glitch`, `invert`, `red-channel`, `edge`, `blur`, `cartoon`, `rainbow-wave`
- **Mode FOTO** — viewfinder di kotak tangan → hitung mundur 3-2-1 → flash → crop kotak tangan (+ filter aktif) → kartu polaroid masuk panel kanan (3 slot, animasi development, tersimpan juga sebagai PNG).
- Gestur tangan: **pinch** (ganti filter), **2 tangan kepal** (toggle 2D/3D), **peace sign** (toggle mode), **buka lebar tangan** (shutter).
- Deteksi multi-tangan (maks 2), smoothing landmark, deteksi di-skip di frame tertentu biar hemat CPU.

## Stack

| Item | Versi |
|---|---|
| Python | 3.11.9 |
| opencv | 5.0.0 |
| mediapipe | 1.0.1 |
| numpy | 2.4.6 |
| OS | Windows (dikembangkan di win32) |

## Instalasi

```bash
git clone https://github.com/<username>/Hand_Tracking.git
cd Hand_Tracking

pip install opencv-python mediapipe numpy
```

Model `models/hand_landmarker.task` (≈7.8 MB) sudah ikut di repo. Kalau file itu hilang, unduh dari [MediaPipe HandLandmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) lalu taruh di folder `models/`.

## Menjalankan

```bash
python hand_traking.py
# atau
python retrolens.py
```

Buka webcam, cari window **"RETROLENS Pake Python"**. Window autosize: 640×480 (mode filter) / 940×480 (mode foto, panel aktif).

> Startup bisa makan waktu ±10 detik (import mediapipe + load model) — itu normal.

## Kontrol

### Gestur tangan

| Gestur | Efek |
|---|---|
| **Pinch** (jempol–kelingking / jempol–telunjuk) | Ganti filter berikutnya |
| **2 tangan kepal** | Toggle portal 2D ↔ 3D |
| **Peace sign** (tahan 1 dtk) | Toggle mode FILTER ↔ FOTO |
| **Buka lebar tangan** (5 jari, tahan 0.6 dtk) | Mulai countdown 3-2-1 (mode FOTO) |

### Keyboard

| Key | Efek |
|---|---|
| `q` | Keluar |
| `n` / `y` | Filter berikutnya / sebelumnya |
| `0` | Filter OFF |
| `c` | Toggle portal 2D / 3D |
| `m` | Toggle mode FILTER ↔ FOTO |
| `s` | Screenshot ke `captures/` |

## Struktur project

```
Hand_Tracking/
├── retrolens.py                 # seluruh logika app (single-file)
├── hand_traking.py              # entry point tipis (nama file typo, biarkan)
├── models/hand_landmarker.task  # model MediaPipe (wajib ada)
├── PRD.md                       # dokumen PRD lengkap (state, pipeline, resep nambah fitur)
├── captures/                    # output screenshot & polaroid (di-gitignore)
└── retrolens.log                # log runtime, truncate tiap start (di-gitignore)
```

## Debugging

Semua aktivitas ditulis ke `retrolens.log` (truncate tiap start): heartbeat per stage, STATS fps/detect/render tiap 3 detik, SLOW/STALL warning, dan traceback kalau crash. Cek baris terakhir di log buat nyari titik macet.

Baseline sehat: **fps ≈ 17, detect ≈ 14ms, render ≈ 3–9ms, show ≈ 10ms**.

## Kontribusi

Baca `PRD.md` dulu sebelum nambah fitur — di situ ada peta kode, state management, dan checklist regresi tiap ubahan.

## Lisensi

Bebas dipakai buat belajar.
