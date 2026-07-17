// main.cpp
//
// KCFTracker (kcftracker.hpp/.cpp) ile TrackingFailureDetector
// (TrackingFailureDetector.h/.cpp) modullerini bir arada kullanan ornek
// uygulama. Akis:
//
//   1) Kameradan/videodan ilk kare okunur, kullanici ROI secer.
//   2) KCFTracker o ROI ile init() edilir.
//   3) Her yeni karede (durum LOST DEGILSE):
//        - tracker.locate(frame)      -> guncel ROI (henuz model EGITILMEZ)
//        - tracker.getLastResponse()  -> KCF'in response map'i (CV_32FC1)
//        - tracker.getLastPsr()       -> PSR
//        - detector.Update(frame, roi, responseMap, psr) -> "Tracking OK/Lost"
//        - confidence yeterince yuksekse tracker.adapt(frame) ile model
//          egitilir; dusuk guvende (occlusion/benzer nesne) adaptasyon
//          atlanir, boylece model bozuk gorunume "ogrenip" drift etmez.
//      Durum LOST ise tracker.locate() hic cagrilmaz, ROI son bilinen
//      konumunda donar: aksi halde artik guncellenmeyen (bayat) bir
//      filtreyle gurultuye kilitlenip ROI'yi cerceve disina surukleyebilir.
//   4) Durum (TRACKING/SUSPECT/LOST) ve confidence ekrana yazilir; kutunun
//      rengi duruma gore degisir. 'r' tusu ile hedef yeniden secilebilir,
//      ESC ile cikilir.
//   5) (yeni eklendi) Occlusion testi: 'o' tusu fareyle gezdirilen siyah bir
//      dikdortgeni acar/kapatir ('+'/'-' boyut). Dikdortgen kareye tracker
//      calismadan ONCE boyandigi icin KCF gercekten ortulmus goruntu gorur;
//      hedefin ustune getirerek kirilma tespitinin davranisi izlenebilir.
//
// Not: Bu dosya, orijinal KCFcpp (joaofaro/KCFcpp) projesinin geri kalan
// dosyalarina (tracker.h, ffttools.hpp, recttools.hpp, fhog.hpp, labdata.hpp)
// ihtiyac duyar. Bu depoya yalnizca kullanicinin yukledigi 4 dosya + bu
// main.cpp eklenmistir; KCFcpp'nin destek dosyalari ayrica projeye dahil
// edilmelidir (bkz. README.md).

#include "kcftracker.hpp"
#include "TrackingFailureDetector.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>
#include <sstream>

namespace
{
    // (yeni eklendi) Fareyle gezdirilen siyah "occluder" dikdortgeni:
    // takip edilen nesnenin uzerine getirilerek occlusion aninda takip
    // kirilmasi tespitinin basarimi test edilir. 'o' ile ac/kapat,
    // '+'/'-' ile boyut degistir. Dikdortgen kareye TRACKER'DAN ONCE
    // boyanir; boylece KCF ve detector gercekten ortulmus goruntuyu gorur.
    struct OccluderState
    {
        cv::Point pos;
        bool      enabled;
        int       halfSize;

        OccluderState() : pos(-1, -1), enabled(false), halfSize(60) {}
    };

    void OnMouse(int event, int x, int y, int /*flags*/, void* userdata)
    {
        OccluderState* state = static_cast<OccluderState*>(userdata);
        if (event == cv::EVENT_MOUSEMOVE)
        {
            state->pos = cv::Point(x, y);
        }
    }

    // Kareyi tracker gormeden once fiziksel olarak karartir.
    void ApplyOccluder(cv::Mat& frame, const OccluderState& state)
    {
        if (!state.enabled)
        {
            return;
        }
        // (yeni eklendi) Fare henuz pencere uzerinde hic hareket etmediyse
        // (pos = -1,-1) dikdortgen gorunmez kaliyordu; o durumda kare
        // ortasinda baslat ki 'o' basilir basilmaz gorunsun.
        cv::Point center = state.pos;
        if (center.x < 0)
        {
            center = cv::Point(frame.cols / 2, frame.rows / 2);
        }
        cv::Rect occ(center.x - state.halfSize,
                     center.y - state.halfSize,
                     state.halfSize * 2, state.halfSize * 2);
        occ &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (occ.width > 0 && occ.height > 0)
        {
            frame(occ).setTo(cv::Scalar::all(0));
            // (yeni eklendi) siyah/koyu sahnede de secilebilsin diye ince
            // gri cerceve (tracker bunu da gorur ama 1px'lik etkisi ihmal
            // edilebilir).
            cv::rectangle(frame, occ, cv::Scalar::all(128), 1);
        }
    }

    cv::Rect SelectTarget(const cv::Mat& frame, const std::string& windowName)
    {
        cv::Rect roi = cv::selectROI(windowName, frame, false, false);
        return roi;
    }

    cv::Scalar ColorForState(TrackingFailureDetector::State state)
    {
        switch (state)
        {
            case TrackingFailureDetector::State::TRACKING: return cv::Scalar(0, 200, 0);   // yesil
            case TrackingFailureDetector::State::SUSPECT:  return cv::Scalar(0, 165, 255); // turuncu
            case TrackingFailureDetector::State::LOST:     return cv::Scalar(0, 0, 255);   // kirmizi
        }
        return cv::Scalar(255, 255, 255);
    }
} // anonim namespace

int main(int argc, char** argv)
{
    // argv[1] verilirse video dosyasi, verilmezse varsayilan kamera (0) acilir.
    cv::VideoCapture capture;
    if (argc > 1)
    {
        capture.open(argv[1]);
    }
    else
    {
        capture.open(0);
    }

    if (!capture.isOpened())
    {
        std::cerr << "Video/kamera acilamadi.\n";
        return 1;
    }

    const std::string windowName = "KCF + TrackingFailureDetector";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    // (yeni eklendi) occluder fare takibi. Dikkat: cv::selectROI kendi fare
    // callback'ini kurup bizimkini ezdigi icin her SelectTarget cagrisindan
    // sonra setMouseCallback yeniden cagrilir.
    OccluderState occluder;
    cv::setMouseCallback(windowName, OnMouse, &occluder);

    cv::Mat frame;
    capture >> frame;
    if (frame.empty())
    {
        std::cerr << "Ilk kare okunamadi.\n";
        return 1;
    }

    cv::Rect roi = SelectTarget(frame, windowName);
    cv::setMouseCallback(windowName, OnMouse, &occluder); // (yeni eklendi) selectROI callback'i ezdi, geri kur
    if (roi.width <= 0 || roi.height <= 0)
    {
        std::cerr << "Gecersiz ROI secildi, cikiliyor.\n";
        return 1;
    }

    // hog=true, fixed_window=true, multiscale=true. Lab renk ozelligi
    // yalnizca 3 kanalli (renkli) girdide acilir; gri/IR kaynakta Lab
    // donusumu (cvtColor CV_BGR2Lab) 1 kanalli goruntude patlar.
    const bool useLab = (frame.channels() == 3);
    KCFTracker tracker(true, true, true, useLab);
    tracker.init(roi, frame);

    TrackingFailureDetector detector;
    cv::Rect trackedRoi = roi; // son bilinen ROI; LOST'ta bu deger korunur

    while (true)
    {
        capture >> frame;
        if (frame.empty())
        {
            break;
        }

        // (yeni eklendi) Occluder'i kareye TRACKER'DAN ONCE boya: KCF ve
        // detector ortulmus goruntuyu gormeli ki gercek occlusion testi olsun.
        ApplyOccluder(frame, occluder);

        bool trackingOk = true;

        // LOST durumundayken KCF'i calistirmaya devam etmiyoruz: adapt()
        // zaten atlandigi icin model bayatlamis oluyor, boyle bir modelle
        // "en iyi eslesmeyi" aramaya devam etmek ROI'yi gurultuye kilitleyip
        // cerceve disina surukleyebiliyor (RectTools::subwindow() icinde
        // assert(0)). LOST'ta ROI donar, kullanici 'r' ile yeniden
        // secene kadar beklenir.
        if (detector.GetState() != TrackingFailureDetector::State::LOST)
        {
            // 1) KCF ile konumu bul (henuz modeli EGITME).
            trackedRoi = tracker.locate(frame);

            // 2) KCF'in bu karedeki response map + PSR'ini detector'a besle.
            trackingOk = detector.Update(frame,
                                          trackedRoi,
                                          tracker.getLastResponse(),
                                          tracker.getLastPsr());

            // 3) Akilli guncelleme: modeli SADECE confidence yeterince
            // yuksekken egit. Dusuk guvende (occlusion / benzer nesne)
            // adaptasyonu atlamak, modelin bozuk gorunume "ogrenip" drift
            // etmesini engeller.
            if (detector.GetConfidence() >= tfd_config::REFERENCE_UPDATE_MIN_CONFIDENCE)
            {
                tracker.adapt(frame);
            }
        }
        else // LOST: locate/adapt cagrilmaz, ROI donuk kalir
        {
            trackingOk = false;
        }

        // 4) Gorsellestirme: durum rengine gore kutu + bilgi metni.
        const cv::Scalar color = ColorForState(detector.GetState());
        cv::rectangle(frame, trackedRoi, color, 2);

        std::ostringstream info;
        info << detector.GetStateName()
             << " | conf=" << static_cast<int>(detector.GetConfidence())
             << " | psr=" << static_cast<int>(tracker.getLastPsr());
        if (occluder.enabled) // (yeni eklendi)
        {
            info << " | OCC " << occluder.halfSize * 2 << "px";
        }
        cv::putText(frame, info.str(), cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);

        if (!trackingOk)
        {
            cv::putText(frame, "TRACKING LOST - 'r' ile yeniden secin",
                        cv::Point(10, 55), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow(windowName, frame);

        const int key = cv::waitKey(1) & 0xFF;
        if (key == 27) // ESC
        {
            break;
        }
        if (key == 'o' || key == 'O') // (yeni eklendi) occluder ac/kapat
        {
            occluder.enabled = !occluder.enabled;
        }
        if (key == '+' || key == '=') // (yeni eklendi) occluder buyut
        {
            occluder.halfSize = std::min(occluder.halfSize + 10, 300);
        }
        if (key == '-' || key == '_') // (yeni eklendi) occluder kucult
        {
            occluder.halfSize = std::max(occluder.halfSize - 10, 10);
        }
        if (key == 'r' || key == 'R')
        {
            // Hedefi yeniden kilitle: KCF'i ve detector'i sifirdan baslat.
            const cv::Rect newRoi = SelectTarget(frame, windowName);
            cv::setMouseCallback(windowName, OnMouse, &occluder); // (yeni eklendi) callback'i geri kur
            if (newRoi.width > 0 && newRoi.height > 0)
            {
                tracker = KCFTracker(true, true, true, frame.channels() == 3);
                tracker.init(newRoi, frame);
                detector.Reset();
                trackedRoi = newRoi; // kutu hemen yeni secime atlasin
            }
        }
    }

    return 0;
}
