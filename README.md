# KCF + TrackingFailureDetector Demo

Bu depo, [joaofaro/KCFcpp](https://github.com/joaofaro/KCFcpp) tabanli bir
KCF takipcisini (`kcftracker.hpp` / `kcftracker.cpp`) hibrit bir takip
kirilmasi tespit modulu (`TrackingFailureDetector.h` / `.cpp`) ile birlestiren
ornek bir entegrasyon sunar.

## Dosyalar

- `src/kcftracker.hpp`, `src/kcftracker.cpp` — KCF takipcisi. Orijinal
  KCFcpp koduna, `TrackingFailureDetector`'in ihtiyac duydugu response map ve
  PSR (Peak-to-Sidelobe Ratio) degerlerini disari acan kucuk bir ek yapildi:
  `getLastResponse()` ve `getLastPsr()`.
- `src/TrackingFailureDetector.h`, `src/TrackingFailureDetector.cpp` — Response
  map + gorunum (histogram/template/NCC/SSIM) metriklerini birlestirip
  0-100 "tracking confidence" skoru ve TRACKING/SUSPECT/LOST durum makinesi
  ureten modul.
- `src/main.cpp` — Ikisini bir arada kullanan ornek uygulama: kameradan/
  videodan okur, ROI secilir, her karede KCF ile takip edilip
  `TrackingFailureDetector` ile dogrulanir; duruma gore kutu rengi ve
  confidence skoru gosterilir.

## Eksik bagimliliklar

`kcftracker.cpp`, orijinal KCFcpp projesinin su destek dosyalarina ihtiyac
duyar (bu depoya dahil edilmemistir, kullanicinin yalnizca `kcftracker.hpp`
ve `kcftracker.cpp` dosyalarini paylasmasi nedeniyle):

- `tracker.h` (soyut `Tracker` taban sinifi)
- `ffttools.hpp`
- `recttools.hpp`
- `fhog.hpp`
- `labdata.hpp`

Bu dosyalari [joaofaro/KCFcpp](https://github.com/joaofaro/KCFcpp)
deposundan alip `src/` klasorune eklemeden proje derlenmez.

## Derleme

```bash
mkdir build && cd build
cmake ..
make
./kcf_demo [video_dosyasi]   # video_dosyasi verilmezse varsayilan kamera (0) acilir
```

## Kullanim

- Ilk karede fare ile hedefi secip Enter/Space'e basin.
- ESC ile cikin, `r` ile hedefi yeniden secin (KCF ve detector sifirlanir).
- Ekranda durum (TRACKING yesil / SUSPECT turuncu / LOST kirmizi),
  confidence ve PSR degerleri gosterilir.
