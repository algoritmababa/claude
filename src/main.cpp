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
//      rengi duruma gore degisir. ESC ile cikilir.
//
// Not: kcftracker, KCFcpp destek dosyalarina (tracker.h, ffttools.hpp,
// recttools.hpp, fhog.hpp, labdata.hpp) ve TrackingFailureDetector.h/.cpp'ye
// ihtiyac duyar; hepsi bu depoda src/ altindadir.

#include "kcftracker.hpp"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <sstream>

namespace
{
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

    cv::Mat frame;
    capture >> frame;
    if (frame.empty())
    {
        std::cerr << "Ilk kare okunamadi.\n";
        return 1;
    }

    cv::Rect roi = SelectTarget(frame, windowName);
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
        cv::putText(frame, info.str(), cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);

        if (!trackingOk)
        {
            cv::putText(frame, "TRACKING LOST",
                        cv::Point(10, 55), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow(windowName, frame);

        const int key = cv::waitKey(1) & 0xFF;
        if (key == 27) // ESC
        {
            break;
        }
    }

    return 0;
}
