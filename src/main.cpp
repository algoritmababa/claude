// main.cpp
//
// KCFTracker (kcftracker.hpp/.cpp) ile TrackingFailureDetector
// (TrackingFailureDetector.h/.cpp) modullerini bir arada kullanan ornek
// uygulama. Akis:
//
//   1) Kameradan/videodan ilk kare okunur, kullanici ROI secer.
//   2) KCFTracker o ROI ile init() edilir.
//   3) Her yeni karede:
//        - tracker.locate(frame)      -> guncel ROI (henuz model EGITILMEZ)
//        - tracker.getLastResponse()  -> KCF'in response map'i (CV_32FC1)
//        - tracker.getLastPsr()       -> PSR
//        - detector.Update(frame, roi, responseMap, psr) -> "Tracking OK/Lost"
//        - confidence yeterince yuksekse tracker.adapt(frame) ile model
//          egitilir; dusuk guvende (occlusion/benzer nesne) adaptasyon
//          atlanir, boylece model bozuk gorunume "ogrenip" drift etmez.
//   4) Durum (TRACKING/SUSPECT/LOST) ve confidence ekrana yazilir; kutunun
//      rengi duruma gore degisir. 'r' tusu ile hedef yeniden secilebilir,
//      ESC ile cikilir.
//
// Not: Bu dosya, orijinal KCFcpp (joaofaro/KCFcpp) projesinin geri kalan
// dosyalarina (tracker.h, ffttools.hpp, recttools.hpp, fhog.hpp, labdata.hpp)
// ihtiyac duyar. Bu depoya yalnizca kullanicinin yukledigi 4 dosya + bu
// main.cpp eklenmistir; KCFcpp'nin destek dosyalari ayrica projeye dahil
// edilmelidir (bkz. README.md).

#include "kcftracker.hpp"
#include "TrackingFailureDetector.h"

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

    // hog=true, fixed_window=true, multiscale=true. Lab renk ozelligi yalnizca
    // 3 kanalli (renkli) girdide acilir; gri/IR kaynakta Lab donusumu
    // (cvtColor CV_BGR2Lab) 1 kanalli goruntude patlar.
    const bool useLab = (frame.channels() == 3);
    KCFTracker tracker(true, true, true, useLab);
    tracker.init(roi, frame);

    TrackingFailureDetector detector;

    while (true)
    {
        capture >> frame;
        if (frame.empty())
        {
            break;
        }

        // 1) KCF ile konumu bul (henuz modeli EGITME).
        const cv::Rect trackedRoi = tracker.locate(frame);

        // 2) KCF'in bu karedeki response map + PSR'ini detector'a besle.
        const bool trackingOk = detector.Update(frame,
                                                 trackedRoi,
                                                 tracker.getLastResponse(),
                                                 tracker.getLastPsr());

        // 3) Akilli guncelleme: modeli SADECE confidence yeterince
        // yuksekken egit. Dusuk guvende (occlusion / benzer nesne / LOST)
        // adaptasyonu atlamak, modelin bozuk gorunume "ogrenip" drift
        // etmesini engeller.
        if (detector.GetConfidence() >= tfd_config::REFERENCE_UPDATE_MIN_CONFIDENCE)
        {
            tracker.adapt(frame);
        }

        // 4) Gorsellestirme: durum rengine gore kutu + bilgi metni.
        const cv::Scalar color = ColorForState(detector.GetState());
        cv::rectangle(frame, trackedRoi, color, 2);

        std::ostringstream info;
        info << detector.GetStateName()
             << " | conf=" << static_cast<int>(detector.GetConfidence())
             << " | psr=" << static_cast<int>(tracker.getLastPsr());
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
        if (key == 'r' || key == 'R')
        {
            // Hedefi yeniden kilitle: KCF'i ve detector'i sifirdan baslat.
            const cv::Rect newRoi = SelectTarget(frame, windowName);
            if (newRoi.width > 0 && newRoi.height > 0)
            {
                tracker = KCFTracker(true, true, true, frame.channels() == 3);
                tracker.init(newRoi, frame);
                detector.Reset();
            }
        }
    }

    return 0;
}
