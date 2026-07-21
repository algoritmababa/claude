#ifndef TRACKING_FAILURE_DETECTOR_H
#define TRACKING_FAILURE_DETECTOR_H

//==============================================================================
// TrackingFailureDetector
//------------------------------------------------------------------------------
// KCF (joaofaro/KCFcpp) tabanli takip icin hibrit takip-kirilmasi tespit modulu.
//
// Girdi : frame, roi, KCF response map (detect() icindeki 'res', CV_32FC1), PSR
// Cikti : bool  -> true  = "Tracking OK"
//                  false = "Tracking Lost"
//
// Tasarim ilkeleri:
//  - OpenCV 3.4 disinda bagimlilik yok, C++11 ile derlenir.
//  - Tum metrikler sabit boyutlu (PATCH_SIZE x PATCH_SIZE) patch uzerinde
//    hesaplanir -> ROI boyutundan bagimsiz, sinirli ve deterministik maliyet.
//  - Response map metrikleri KCF'in zaten urettigi harita uzerinde O(N) gecis.
//  - Referanslar (template + histogram) yalnizca TRACKING durumunda ve yuksek
//    guvende EMA ile guncellenir -> drift'e karsi korumali.
//  - Tek karede LOST'a gecilmez; histerezisli durum makinesi kullanilir.
//==============================================================================

#include <opencv2/core.hpp>

//------------------------------------------------------------------------------
// Tum esikler, agirliklar ve sabitler tek yerde. Magic number yok.
// C++11: namespace-scope constexpr degiskenler header'da guvenlidir
// (const -> internal linkage).
//------------------------------------------------------------------------------
namespace tfd_config
{
    //--- Patch / on-isleme ---------------------------------------------------
    constexpr int    PATCH_SIZE                 = 64;    // tum benzerlik metrikleri bu boyutta
    constexpr int    MIN_VALID_ROI_SIZE         = 8;     // px; daha kucuk ROI gecersiz sayilir
    constexpr double NUMERIC_EPSILON            = 1e-9;

    //--- Response map metrik parametreleri ------------------------------------
    // Ikincil tepe aranirken ana tepe cevresinde maskelenecek yaricap,
    // harita kisa kenarinin orani olarak (kucuk haritalarda alt sinir uygulanir).
    constexpr double SECOND_PEAK_EXCLUSION_RATIO = 0.15;
    constexpr int    SECOND_PEAK_EXCLUSION_MIN   = 2;    // px
    constexpr int    SHARPNESS_WINDOW_RADIUS     = 2;    // 5x5 komsuluk

    //--- Histogram parametreleri ----------------------------------------------
    constexpr int    HIST_HUE_BINS              = 30;
    constexpr int    HIST_SAT_BINS              = 32;
    constexpr int    HIST_GRAY_BINS             = 64;
    constexpr float  HIST_HUE_RANGE_MAX         = 180.0f;
    constexpr float  HIST_SAT_RANGE_MAX         = 256.0f;
    constexpr float  HIST_GRAY_RANGE_MAX        = 256.0f;

    //--- SSIM sabitleri (Wang et al., L = 255) ---------------------------------
    constexpr double SSIM_K1                    = 0.01;
    constexpr double SSIM_K2                    = 0.03;
    constexpr double SSIM_DYNAMIC_RANGE         = 255.0;
    constexpr double SSIM_C1 = (SSIM_K1 * SSIM_DYNAMIC_RANGE) * (SSIM_K1 * SSIM_DYNAMIC_RANGE);
    constexpr double SSIM_C2 = (SSIM_K2 * SSIM_DYNAMIC_RANGE) * (SSIM_K2 * SSIM_DYNAMIC_RANGE);
    constexpr int    SSIM_GAUSS_KERNEL          = 11;
    constexpr double SSIM_GAUSS_SIGMA           = 1.5;

    //--- Normalizasyon rampalari [min, max] -> [0, 1] --------------------------
    // Degerler KCF'in tipik calisma araligina gore secilmistir; CSV loglariyla
    // gercek sekanslar uzerinde kalibre edilmelidir.
    constexpr double PSR_NORM_MIN               = 4.0;
    constexpr double PSR_NORM_MAX               = 20.0;
    constexpr double PEAK_VALUE_NORM_MIN        = 0.20;
    constexpr double PEAK_VALUE_NORM_MAX        = 0.90;
    constexpr double APCE_NORM_MIN              = 10.0;
    constexpr double APCE_NORM_MAX              = 50.0;
    constexpr double PEAK_RATIO_NORM_MIN        = 0.05;  // (1 - ikincil/ana) icin
    constexpr double PEAK_RATIO_NORM_MAX        = 0.50;
    constexpr double SHARPNESS_NORM_MIN         = 0.05;
    constexpr double SHARPNESS_NORM_MAX         = 0.50;
    constexpr double HIST_SIM_NORM_MIN          = 0.40;
    constexpr double HIST_SIM_NORM_MAX          = 0.90;
    constexpr double TEMPLATE_SIM_NORM_MIN      = 0.50;
    constexpr double TEMPLATE_SIM_NORM_MAX      = 0.90;
    constexpr double NCC_NORM_MIN               = 0.40;
    constexpr double NCC_NORM_MAX               = 0.90;
    constexpr double SSIM_NORM_MIN              = 0.30;
    constexpr double SSIM_NORM_MAX              = 0.80;

    //--- Confidence agirliklari (toplam = 1.0) ---------------------------------
    // Response map grubu (0.60): aninda, kare-ici kanit.
    // Gorunum grubu (0.40): occlusion / benzer nesne / drift'e karsi kanit.
    constexpr double WEIGHT_PSR                 = 0.20;
    constexpr double WEIGHT_PEAK_VALUE          = 0.10;
    constexpr double WEIGHT_DISTRIBUTION        = 0.15;  // APCE
    constexpr double WEIGHT_PEAK_RATIO          = 0.10;
    constexpr double WEIGHT_SHARPNESS           = 0.05;
    constexpr double WEIGHT_HIST_SIM            = 0.10;
    constexpr double WEIGHT_TEMPLATE_SIM        = 0.15;
    constexpr double WEIGHT_NCC                 = 0.05;  // ilk template'e gore drift dedektoru
    constexpr double WEIGHT_SSIM                = 0.10;

    //--- Confidence olcegi ve zamansal filtre -----------------------------------
    constexpr double CONFIDENCE_MAX             = 100.0;
    constexpr double EMA_ALPHA                  = 0.30;  // yuksek -> hizli tepki
    constexpr double INITIAL_CONFIDENCE         = 80.0;  // init sonrasi baslangic

    //--- Durum makinesi esikleri (histerezis) ----------------------------------
    constexpr double CONF_TRACKING_THRESHOLD    = 60.0;  // SUSPECT/LOST -> TRACKING icin
    constexpr double CONF_SUSPECT_THRESHOLD     = 45.0;  // TRACKING -> SUSPECT altina dusunce
    constexpr int    LOST_CONSECUTIVE_FRAMES    = 8;     // SUSPECT'te ardisik dusuk kare -> LOST
    constexpr int    RECOVER_FRAMES_FROM_SUSPECT = 3;    // ardisik yuksek kare -> TRACKING
    constexpr int    RECOVER_FRAMES_FROM_LOST    = 5;

    //--- Referans guncelleme ----------------------------------------------------
    constexpr double REFERENCE_UPDATE_MIN_CONFIDENCE = 70.0;
    constexpr double REFERENCE_EMA_ALPHA             = 0.02; // yavas adaptasyon
} // namespace tfd_config

//------------------------------------------------------------------------------
class TrackingFailureDetector
{
public:
    enum class State
    {
        TRACKING,
        SUSPECT,
        LOST
    };

    // Ham + normalize metrikler; CSV loglama / kalibrasyon icin disari acilir.
    struct Metrics
    {
        // Ham degerler
        double psr;
        double peakValue;
        double peakRatio;            // 1 - (ikincilTepe / anaTepe)
        double peakSharpness;
        double responseDistribution; // APCE
        double histSimilarity;
        double templateSimilarity;
        double ncc;
        double ssim;

        // Normalize [0,1] degerler
        double nPsr;
        double nPeakValue;
        double nPeakRatio;
        double nPeakSharpness;
        double nResponseDistribution;
        double nHistSimilarity;
        double nTemplateSimilarity;
        double nNcc;
        double nSsim;

        Metrics();
    };

    TrackingFailureDetector();

    // Ana giris noktasi. Her karede, KCF update() sonrasi cagrilir.
    //   frame       : mevcut kare (CV_8UC1 veya CV_8UC3)
    //   roi         : KCF'in dondurdugu guncel ROI
    //   responseMap : KCF detect() icindeki response map (CV_32FC1)
    //   psr         : mevcut PSR degeri
    // Donus: true = "Tracking OK", false = "Tracking Lost"
    bool Update(const cv::Mat& frame,
                const cv::Rect& roi,
                const cv::Mat& responseMap,
                double psr);

    // Yeni hedef kilitlendiginde (KCF init ile birlikte) cagrilmalidir.
    void Reset();

    // Gozlem / loglama arayuzu
    double         GetConfidence()         const { return filteredConfidence_; }
    double         GetRawConfidence()      const { return rawConfidence_;      }
    State          GetState()              const { return state_;              }
    const Metrics& GetLastMetrics()        const { return lastMetrics_;        }
    const char*    GetStateName()          const;

private:
    //--- Response map metrikleri -------------------------------------------
    static double CalculatePeakValue(const cv::Mat& responseMap,
                                     cv::Point& peakLocation);
    static double CalculatePeakRatio(const cv::Mat& responseMap,
                                     const cv::Point& peakLocation,
                                     double peakValue);
    static double CalculatePeakSharpness(const cv::Mat& responseMap,
                                         const cv::Point& peakLocation,
                                         double peakValue);
    static double CalculateResponseDistribution(const cv::Mat& responseMap);

    //--- Gorunum (appearance) metrikleri ------------------------------------
    double CalculateHistogramSimilarity(const cv::Mat& patchForHist) const;
    double CalculateTemplateSimilarity(const cv::Mat& grayPatch32F) const;
    double CalculateNCC(const cv::Mat& grayPatch32F) const;
    static double CalculateSSIM(const cv::Mat& img1_32F,
                                const cv::Mat& img2_32F);

    //--- Karar zinciri --------------------------------------------------------
    static void   NormalizeMetrics(Metrics& metrics);
    static double CalculateConfidence(const Metrics& metrics);
    bool          UpdateStateMachine(double filteredConfidence);

    //--- Yardimcilar ----------------------------------------------------------
    static double NormalizeLinear(double value, double rangeMin, double rangeMax);
    static bool   ExtractPatches(const cv::Mat& frame,
                                 const cv::Rect& roi,
                                 cv::Mat& patchForHist,
                                 cv::Mat& grayPatch32F);
    cv::Mat       ComputeHistogram(const cv::Mat& patchForHist) const;
    void          InitializeReferences(const cv::Mat& patchForHist,
                                       const cv::Mat& grayPatch32F);
    void          UpdateReferences(const cv::Mat& patchForHist,
                                   const cv::Mat& grayPatch32F);
    void          ApplyLowConfidenceFrame(); // gecersiz girdi yolu

    //--- Durum ------------------------------------------------------------------
    State   state_;
    bool    initialized_;
    bool    useColorHistogram_;      // frame kanal sayisina gore init'te belirlenir

    double  rawConfidence_;
    double  filteredConfidence_;
    int     lowConfidenceCounter_;
    int     highConfidenceCounter_;

    cv::Mat referenceTemplate_;      // EMA ile guncellenen gri template (CV_32FC1)
    cv::Mat initialTemplate_;        // kilitlenme anindaki template (drift referansi)
    cv::Mat referenceHistogram_;

    Metrics lastMetrics_;
};

#endif // TRACKING_FAILURE_DETECTOR_H
