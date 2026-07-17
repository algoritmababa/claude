// main.cpp
//
// KCFTracker kullanan ornek uygulama. Takip-kirilmasi tespiti
// (TrackingFailureDetector) artik KCFTracker'in ICINE gomulu; main yalnizca
// kcftracker.hpp API'siyle konusur. Akis:
//
//   1) Kameradan/videodan ilk kare okunur, kullanici ROI secer.
//   2) KCFTracker o ROI ile init() edilir (gomulu dedektor da sifirlanir).
//   3) Her yeni karede tracker.update(frame) cagrilir; icerde:
//        - LOST degilse: locate() ile konum bulunur, response map + PSR
//          gomulu dedektore beslenir, confidence yeterliyse adapt() ile
//          model egitilir (dusuk guvende adaptasyon atlanir -> drift yok).
//        - LOST ise: locate/adapt hic calismaz, son ROI donuk doner.
//      Sonuclar tracker.isTrackingOk() / getConfidence() /
//      getTrackState(Name)() ile okunur.
//   4) Durum (TRACKING/SUSPECT/LOST) ve confidence ekrana yazilir; kutunun
//      rengi duruma gore degisir. 'r' tusu ile hedef yeniden secilebilir,
//      ESC ile cikilir.
//   5) (yeni eklendi) Occlusion testi: 'o' tusu fareyle gezdirilen siyah bir
//      dikdortgeni acar/kapatir ('+'/'-' boyut). Dikdortgen kareye tracker
//      calismadan ONCE boyandigi icin KCF gercekten ortulmus goruntu gorur;
//      hedefin ustune getirerek kirilma tespitinin davranisi izlenebilir.
//
// Not: kcftracker, KCFcpp destek dosyalarina (tracker.h, ffttools.hpp,
// recttools.hpp, fhog.hpp, labdata.hpp) ve TrackingFailureDetector.h/.cpp'ye
// ihtiyac duyar; hepsi bu depoda src/ altindadir.

#include "kcftracker.hpp"

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

        // Takip + kirilma tespiti + kosullu model egitimi tek cagrida:
        // hepsi KCFTracker::update() icinde yurur. LOST durumunda update()
        // son ROI'yi donuk dondurur (locate/adapt calismaz).
        trackedRoi = tracker.update(frame);
        const bool trackingOk = tracker.isTrackingOk();

        // Gorsellestirme: durum rengine gore kutu + bilgi metni.
        const cv::Scalar color = ColorForState(tracker.getTrackState());
        cv::rectangle(frame, trackedRoi, color, 2);

        std::ostringstream info;
        info << tracker.getTrackStateName()
             << " | conf=" << static_cast<int>(tracker.getConfidence())
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
            // Hedefi yeniden kilitle: init() gomulu dedektoru de sifirlar.
            const cv::Rect newRoi = SelectTarget(frame, windowName);
            cv::setMouseCallback(windowName, OnMouse, &occluder); // (yeni eklendi) callback'i geri kur
            if (newRoi.width > 0 && newRoi.height > 0)
            {
                tracker = KCFTracker(true, true, true, frame.channels() == 3);
                tracker.init(newRoi, frame);
                trackedRoi = newRoi; // kutu hemen yeni secime atlasin
            }
        }
    }

    return 0;
}
