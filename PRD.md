# PRD — RETROLENS

> Dokumen hidup. **Update file ini tiap kali nambah/mengubah fitur.**
> Versi dokumen: 3 Oktober 2026 (fitur FOTO + panel polaroid sudah masuk).
> Anchor kode pakai nama fungsi/konstanta (bukan nomor baris — baris bergeser tiap edit).

---

## 1. Ringkasan

RETROLENS adalah app webcam **hand-tracking** (Python + MediaPipe + OpenCV): tangan lo jadi "portal" yang nampilin efek filter, plus sekarang ada **mode foto** dengan hasil polaroid di panel samping.

**Dua app mode:**

| Mode | Isi |
|---|---|
| `FILTER` | Portal filter (default). Ada sub-mode portal **2D** (hull tangan) dan **3D** (kubus mengambang). |
| `FOTO` | Viewfinder di kotak tangan → hitung mundur 3-2-1 → flash → crop kotak tangan (+filter aktif) → polaroid masuk panel kanan. |

**Prinsip:** semua fitur lama wajib tetap jalan apa adanya (regresi nol). Gestur/keyboard lama tidak boleh dipakai ulang untuk hal baru tanpa mengganti state-nya.

---

## 2. Stack & struktur file

| Item | Nilai |
|---|---|
| Python | 3.11.9 |
| opencv | 5.0.0 |
| mediapipe | 1.0.1 |
| numpy | 2.4.6 |
| OS | Windows (dikembangkan di win32) |

```
Hand_Tracking/
├── retrolens.py          # SELURUH logika app (single-file, ~1050 baris)
├── hand_traking.py       # entry point tipis: from retrolens import main (nama file typo, biarkan)
├── models/hand_landmarker.task   # model MediaPipe (wajib ada)
├── retrolens.log         # di-truncate tiap start (mode "w"), regenerate
└── captures/             # dibuat otomatis saat pertama screenshot/foto
    ├── retrolens_<ts>.png        # screenshot 's'
    └── polaroid_<ts>.png         # hasil foto mode FOTO (kartu polaroid utuh)
```

Jalankan: `python hand_traking.py` (atau `python retrolens.py`).
Window: **"RETROLENS Pake Python"** — 640×480 (mode filter) / 940×480 (mode foto, panel aktif). Window autosize, ikut lebar frame yang di-`imshow`.

---

## 3. Pipeline per frame (loop utama di `main()`)

Thread & lifecycle:

- **CameraThread** — baca frame terus-menerus, `snapshot()` diambil main loop (lock). Biar `cap.read()` nggak bikin loop macet.
- **heartbeat thread** — log `HB stage=<nama>` tiap 1 dtk ke file (echo=False).
- `stage` dict → dipakai buat nyari titik macet dari log (`camera/flip/detect/render/show/keys`).
- `sys.excepthook` → traceback masuk log + `_cleanup()` + `input ENTER`.
- `atexit` → `_cleanup()` (release cap, close landmarker, destroy window).
- Kalau kamera nggak menghasilkan frame >15 dtk → log ERROR + keluar.

Urutan stage dalam satu iterasi:

1. **camera** — `cam.snapshot()`. Frame belum datang / sama → tampilin frame terakhir, `continue`.
2. **flip** — `cv2.flip(raw, 1)` (mirror), hitung FPS EMA.
3. **detect / reuse** — `landmarker.detect_for_video()` **hanya kalau** `prev_lms is None or dt < 0.035 or frames_since_detect >= 1` (di baseline: det≈50%, detect≈14ms). Sisanya pakai hasil deteksi sebelumnya.
4. **render** — deteksi tangan → gesture → portal → filter → capture → gambar (portal/hands/pinch/viewfinder/countdown/HUD/banner).
5. **show** — compose panel polaroid (mode foto) → flash → `imshow` → `waitKey(1)`.
6. **keys** — handling keyboard → STATS log tiap 3 dtk.

Detail penting di stage render:

- **`last_hands` cache 0.5 dtk** — kalau deteksi ilang sesaat, gambar pakai tangan terakhir.
- **urutan gambar**: `filter_in_mask` (frame bersih + filter di region) → **capture pakai frame di titik ini** → `draw_portal` → `draw_hands` → `draw_pinch_indicator` → viewfinder → countdown → HUD → banner.
  ⚠️ **Capture harus SEBELUM garis portal/hands/HUD digambar**, biar nggak kebakar ke foto.

---

## 4. State yang ada (yang harus dipahami sebelum nambah apa pun)

| State | Nilai | Fungsi |
|---|---|---|
| `app_mode` | `APP_FILTER` / `APP_PHOTO` | App mode utama (filter vs foto). |
| `mode` | `"2d"` / `"3d"` | Sub-mode portal — **hanya berlaku di mode filter**. Mode foto selalu pakai hull 2D (`eff_mode`). |
| `shot` | `idle → countdown → flash → idle` | Mesin state foto. Toggle mode diblok kecuali `shot == "idle"`. |
| `filter_index` + `none_active` | index ke `FILTER_NAMES` / filter OFF | Filter aktif — **berlaku untuk mode filter & mode foto**. |
| `pinch_armed` / `palm_armed` / `peace_armed` | bool | Pola re-arm: trigger sekali, harus lepas gestur dulu biar ke-trigger lagi. |
| `polaroids` | list `Polaroid`, maks `MAX_POLAROIDS` | Di memori (tampil di panel); semua hasil juga disimpan ke PNG. |
| `panel_img` | ndarray / None | Cache panel. Di-reset ke None saat toggle mode. |

---

## 5. Gestur & keyboard (inventaris lengkap)

### Gestur tangan

| Gestur | Deteksi | Efek | Debounce |
|---|---|---|---|
| **Pinch** (jempol–kelingking / jempol–telunjuk) | `pinch_ratio() < PINCH_TOUCH` (0.65) dan bukan kepal; lepas saat `> PINCH_RELEASE` (0.95) | Ganti filter (next). Berlaku di **semua mode, termasuk selama countdown**. | armed + `0.4s` |
| **2 tangan kepal** | `is_fist()` di kedua tangan | Toggle sub-mode portal 2D ↔ 3D (mode filter saja yang kelihatan efeknya). | `1.2s` |
| **Peace sign (V)** | `is_peace()` — telunjuk+middle extended, manis+kelingking lipat | Toggle `app_mode` FILTER ↔ FOTO (`toggle_app_mode()`). Diblok saat countdown; kalau dipaksakan saat countdown, timer hold di-reset. | hold `PEACE_HOLD`=1.0 dtk, cooldown `PEACE_COOLDOWN`=1.5 dtk, harus lepas dulu |
| **Buka lebar tangan (5 jari)** | `is_open_palm()` — ke-4 jari extended + `pinch_ratio > PINCH_RELEASE` | **Hanya di mode FOTO & `shot=="idle"`**: mulai countdown 3-2-1. | hold `PALM_HOLD`=0.6 dtk, edge-triggered (harus nutup tangan dulu setelah tiap tembakan) |

### Keyboard

| Key | Efek |
|---|---|
| `q` | Keluar |
| `n` / `y` (atau `p`) | Filter berikutnya / sebelumnya |
| `0` | Filter OFF (toggle `none_active`) |
| `c` | Toggle sub-mode portal 2D / 3D |
| `m` | Toggle app mode FILTER ↔ FOTO (sama dengan peace). Saat countdown → banner "TUNGGU COUNTDOWN SELESAI DULU". |
| `s` | Screenshot **frame komposit** (feed + panel kalo kelihatan) ke `captures/` |

---

## 6. Filter

Urutan cycle = urutan `FILTER_NAMES` (retrolens.py). Default saat start: `pixelate`.

`none` (OFF) tidak ada di list — lewat `none_active` / key `0`.

| # | Nama | Ide |
|---|---|---|
| 1 | dual-tone | duotone ungu–oranye |
| 2 | thermal | colormap JET |
| 3 | sketch | dodge-burn sketsa |
| 4 | pixelate | block 16px |
| 5 | glitch | geser channel + strip acak |
| 6 | invert | bitwise not |
| 7 | red-channel | cuma merah |
| 8 | edge | Canny menyatu ke frame asli |
| 9 | blur | Gaussian 31×31 |
| 10 | cartoon | bilateral + edge hitam |
| 11 | rainbow-wave | geser hue per kolom animasi |

Filter hanya diterapkan **di dalam mask hull** (`filter_in_mask()`), bukan full frame. `filter_in_mask` menghitung bbox mask → crop → apply → timbal balik. Return `(frame_out, jumlah_px_mask)`.

---

## 7. Fitur FOTO (detail)

### Flow

```
[mode FILTER] --peace tahan 1dtk / 'm'--> [mode FOTO, shot=idle]
    HUD hijau, viewfinder kuning di kotak tangan, panel BELUM ada (belum ada foto)
--> buka lebar tangan 0.6 dtk --> shot=countdown
    angka 3-2-1 besar + progress bar, bracket merah, pinch tetap bisa ganti filter
--> shot=flash (0.22dtk, layar putih fade) + CAPTURE --> shot=idle
    polaroid masuk panel (efek film development) + PNG disimpan
--peace/'m'--> kembali mode FILTER (panel disembunyikan; polaroid tetap di memori)
```

### Aturan capture (`build_polaroid`)

- Sumber: frame **yang sudah di-filter di region hull**, sebelum gambar UI apa pun.
- Area: bbox hull → di-square-in (`side = min(max(bw,bh), H, W)`) → clamp ke tepi frame. Tolak kalau bbox < 40px (banner "FOTO GAGAL").
- Hull nggak ada saat countdown habis → banner "FOTO GAGAL - tangan ilang", flash tetap jalan.
- Filter yang dipakai = `filter_name` **saat frame capture** (bisa berubah selama countdown).

### Kartu polaroid (`make_card` / `Polaroid`)

| Properti | Nilai |
|---|---|
| Kartu | `CARD_W`×`CARD_H` = 150×180, warna (247,245,240) |
| Foto | `PHOTO_SZ` = 120×120, border atas/samping 15px, bawah 45px, outline abu tipis |
| Caption | jam (`%H:%M:%S`) + nama filter, di border bawah |
| Rotasi | acak ±4° per polaroid (di-set saat dibuat) |
| Efek development | fade dari putih (250,249,245) ke asli selama `DEV_DURATION`=1.2 dtk |

### Panel kanan (`render_panel`)

- Ukuran `PHOTO_PANEL_W`×`FRAME_H` = 300×480, bg (26,26,30), di-`np.hstack` ke kanan feed.
- **Muncul hanya jika** `app_mode == APP_PHOTO and len(polaroids) > 0`.
- 3 slot zig-zag (sesuai sketsa gambar tangan): `SLOT_TOP`=[6,150,294] (y kartu), `SLOT_CX`=[170,130,168] (x tengah).
- Polaroid **terbaru selalu slot paling bawah**; slot lama menggeser ke atas (lerp `dt*10`), kartu ke-4 mengeluarkan yang paling lama (`polaroids.pop(0)`).
- Kartu baru masuk dari `y=480` (slide dari bawah). Drop shadow = blur mask alpha ×0.55, offset (+5,+7).
- **Cache**: render cuma saat animasi aktif / `panel_img is None` / transisi animasi→diam. Rotasi kartu di-cache per quantized progress dev (`Polaroid.rotated`).

### Output

- Semua foto (termasuk yang udah keluar dari panel) tersimpan di `captures/polaroid_<epoch_ms>.png` (kartu utuh, tanpa rotasi).

---

## 8. HUD, banner, viewfinder, countdown

**HUD** (`draw_hud`) — 4 baris, teks 0.55, outline hitam tebal 4, baseline y=32 step 28, x=12.

- Mode FILTER: kuning `BANNER_COLOR`; baris info MODE 2D/3D, FILTER, PINCH/portal/FPS, shortcut keyboard.
- Mode FOTO: hijau `PHOTO_COLOR`; baris APP mode, FILTER+hint pinch, hint buka tangan+tangan+FPS, jumlah polaroid+portal+shortcut.
- Filter OFF → baris filter jadi oranye `BANNER_COLOR_OFF`.
- `n_render == 0` (tangan nggak ke-detect) → baris terakhir diganti hint angkat tangan.

**Banner** (`draw_banner`) — teks besar tengah, baseline y=105, dipakai untuk: ganti filter, toggle mode, hasil capture.

**Viewfinder** (`draw_viewfinder`, mode foto saja) — corner bracket L 24px tebal 3 + reticle tengah; kuning saat idle, **merah saat countdown**. Label `PHOTO | <filter>` di **bawah** bbox (`min(H-26, yj+24)`), hint `BUKA TANGAN = SHUTTER` di `max(yi-12, 145)` — angka 145 = di bawah zona HUD, jangan ditaruh di atas (pernah bug: teks nimpa HUD).

**Countdown** (`draw_countdown`) — angka tengah layar, scale `1.7 + 1.1*frac` (pop tiap detik), outline hitam 10, kuning tebal 4 + progress bar 260×10.

Warna: kuning (0,255,255) · hijau foto (0,255,160) · oranye-off (0,100,255) · merah (0,0,255).

---

## 9. Konstanta tunable (semua di atas `CUBE_LOCAL`, satu blok)

| Konstanta | Nilai | Arti kalau diubah |
|---|---|---|
| `FRAME_W/H` | 640×480 | Resolusi kamera yang diminta |
| `PHOTO_PANEL_W` | 300 | Lebar panel; komposit jadi 940px |
| `MAX_POLAROIDS` | 3 | Jumlah polaroid tampil. **Wajib `len(SLOT_*)` ≥ ini** |
| `CARD_W/H`, `PHOTO_SZ` | 150/180/120 | Ukuran kartu; `SLOT_TOP` harus tetap muat (y terakhir + `CARD_H` ≤ 480) |
| `SLOT_TOP` / `SLOT_CX` | [6,150,294] / [170,130,168] | Posisi slot panel (y atas / x tengah) |
| `COUNTDOWN_SEC` | 3 | Durasi hitung mundur |
| `FLASH_DURATION` | 0.22 | Durasi flash putih (detik) |
| `PEACE_HOLD` / `PEACE_COOLDOWN` | 1.0 / 1.5 | Tahan peace utk ganti mode / jeda minimum |
| `PALM_HOLD` | 0.6 | Tahan buka tangan utk mulai countdown |
| `DEV_DURATION` | 1.2 | Efek film development polaroid |
| `PINCH_TOUCH` / `PINCH_RELEASE` | 0.65 / 0.95 | Threshold pinch (rasio terhadap lebar telapak) |
| `TIP_IDS` `PALM_IDS` `PORTAL_IDS` `PORTAL_2D_IDS` | landmark MediaPipe | Titik pembentuk portal |
| `FINGER_PIPS` | (8,6)(12,10)(16,14)(20,18) | Pasangan tip–PIP utk cek jari extended |

---

## 10. Peta kode (retrolens.py)

| Bagian | Fungsi/kelas |
|---|---|
| Util & logging | `log`, `_cleanup`, `_excepthook`, `CameraThread` |
| Filter | `filter_*` (11 buah) → dict `FILTERS` → `apply_filter`, `filter_in_mask` |
| Geometri | `to_px`, `dist`, `convex_hull`, `fill_mask`, `pose_cube_points`, `smooth_corners` |
| Gestur | `is_fist`, `pinch_ratio`, `finger_extended`, `is_peace`, `is_open_palm` |
| Gambar | `draw_hands`, `draw_pinch_indicator`, `draw_portal`, `draw_hud`, `draw_banner` |
| FOTO | `Polaroid`, `layout_polaroids`, `paste_alpha`, `render_panel`, `make_card`, `build_polaroid`, `draw_viewfinder`, `draw_countdown` |
| Entry | `save_screenshot`, `main()` |

---

## 11. Resep nambah fitur

### A. Filter baru
1. Tulis `def filter_nama(frame, t): ...` → return BGR **seukuran input**.
2. Daftarkan ke dict `FILTERS` **dan** tambahkan namanya ke `FILTER_NAMES` (urutan list = urutan cycle pinch).
3. Selesai — otomatis kepakai di portal 2D/3D, mode foto, dan capture.

### B. Gestur baru
1. Buat fungsi murni `is_xxx(pts) -> bool` di blok gestur (pola `is_peace`/`is_open_palm`).
2. Cek bentrok dengan gestur lama (peace/palm/pinch/fist pakai landmark yang sama!).
3. Konsumsi di dalam loop `main()` pakai pola **armed + hold/cooldown + re-arm saat gestur lepas** — jangan cuma `if gestur: aksi` (bakal ke-trigger tiap frame).
4. Kalau gestur butuh "sekali per aksi", tempel ke state `*_armed`.

### C. App mode baru
1. Tambah konstanta `APP_X = "x"` + warna.
2. Masukkan ke kondisi `toggle_app_mode` dan `eff_mode` (kalo pengaruh ke portal).
3. Tambah cabang di `draw_hud` + baris hint.

### D. Ganti jumlah/posisi polaroid
`MAX_POLAROIDS` + `SLOT_TOP` + `SLOT_CX` harus konsisten (lihat §9), dan cek `PHOTO_PANEL_W` masih muat lebar kartu terrotasi (≈ `CARD_W*cos4° + CARD_H*sin4°` + shadow).

### E. Checklist regresi setiap ubahan
1. `python hand_traking.py` jalan >30 dtk tanpa `SLOW/STALL/ERROR` di `retrolens.log`.
2. Mode filter: pinch ganti filter, 2 kepal toggle 2D/3D, `n/y/0/c/q/s` normal.
3. Mode foto full flow: peace → buka tangan → countdown → flash → polaroid masuk panel → PNG ada di `captures/`.
4. Foto ke-4 keluar, panel hilang saat balik mode filter.
5. Perhatiin baris `STATS` (fps/detect/render) — naik-turun drastis = ada masalah performa.

> Baseline sehat (laptop ini): **fps ≈ 17, detect ≈ 14ms, render ≈ 3–9ms, show ≈ 10ms, det ≈ 50%**, `hands=1..2`, startup log muncul ±10 dtk (import mediapipe + load model — normal).

---

## 12. Logging

- `retrolens.log` ditulis tiap start (truncate), line-buffered, satu lock.
- Yang di-log: banner startup + kontrol, ganti mode, countdown, hasil capture (nama file), STATS tiap 3 dtk, HB tiap 1 dtk, SLOW (>1s per stage), CAMERA STALL, traceback kalau crash.
- Pakai log ini dulu buat debug — jarang perlu `print`.

---

## 13. Known issue / catatan

- [ ] **Run pertama kadang aneh**: sekali teramati log berhenti di `HB stage=detect` lalu nggak jalan (app kayak freeze) — re-run berikutnya normal 3× berturut-turut. Kalau kejadian lagi: cek baris terakhir log (`stage` nunjukin titik macet) dan laporkan di sini + tanggal.
- [ ] Nama file `hand_traking.py` typo ("traking") — sengaja dipertahankan biar command lama tetap jalan.
- [ ] Tidak ada unit test di repo (smoke test waktu develop cuma script sementara). Lihat backlog.
- [ ] Window bisa kedip sekali waktu lebar frame berubah 640↔940 (autosize) — kosmetik.

---

## 14. Backlog / ide ke depan

- [ ] Pindahin smoke test (gestur, build_polaroid, panel layout, compose) ke `test_retrolens.py` di repo biar ada regresi otomatis.
- [ ] Tombol hapus polaroid terakhir (gesture atau key, misal `d`).
- [ ] Filter khusus mode foto (list terpisah / ganti filter via gesture selain pinch).
- [ ] Simpan juga full-frame (tanpa crop) saat capture, atau 2 versi.
- [ ] Sound effect klik rana + tick countdown (butuh `winsound`/file wav).
- [ ] Pilihan countdown 0 dtk (instant) via key.
- [ ] Tombol ganti slot/ukuran panel, atau polaroid juga tampil di mode filter (setting).
- [ ] Balik foto (mirror) opsi — crop sekarang pakai frame yang sudah di-flip.
- [ ] Multi-shot burst (gestur tahan → 3 foto beruntun).

---

## 15. Riwayat perubahan

| Tanggal | Perubahan |
|---|---|
| 3 Okt 2026 | Versi awal app: portal filter 2D/3D, 11 filter, gestur pinch/kepal. |
| 3 Okt 2026 | **Fitur FOTO**: app mode filter/foto (peace / `m`), countdown 3-2-1 + flash, capture crop kotak tangan + filter aktif, panel polaroid kanan (3 slot, development effect, animasi slide), viewfinder, PNG ke `captures/`, HUD 2 mode. |
