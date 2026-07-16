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
- `src/tracker.h`, `src/ffttools.hpp`, `src/recttools.hpp`, `src/fhog.hpp`,
  `src/fhog.cpp`, `src/labdata.hpp` — [joaofaro/KCFcpp](https://github.com/joaofaro/KCFcpp)
  deposundan alinan, KCFTracker'in derlenebilmesi icin gereken destek
  dosyalari (degistirilmeden eklendi).

## Lisans notu

Bu depodaki KCF destek dosyalarinin cogu (`ffttools.hpp`, `recttools.hpp`,
`fhog.hpp`/`.cpp`, `kcftracker.hpp`/`.cpp`) BSD 3-clause lisanslidir; tam
metin `LICENSE-KCFcpp` dosyasinda ve ilgili dosyalarin basliklarinda yer
alir.

`src/tracker.h` ise farkli ve daha kisitlayici bir lisans notuyla gelir
(Henriques/Bailer, "sadece arastirma amacli"; ticari veya farkli bir
kullanim icin yazarlardan izin gerektigini, degistirilmis/degistirilmemis
halinin kendi baniza yayinlanmasinin izinsiz oldugunu belirtir). Bu dosya,
kullanicinin acik talimatiyla depoya eklenmistir; production/ticari
kullanimdan once orijinal yazarlardan izin alinmasi onerilir.

## Eksik bagimliliklar

Yukaridaki destek dosyalari bu depoya eklendigi icin proje artik
KCFcpp'nin geri kalanina bagimli degildir; OpenCV disinda ek bir bagimlilik
gerekmez.

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
