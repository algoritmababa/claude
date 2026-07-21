#include "TrackingFailureDetector.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace
{
    using namespace tfd_config;

    // C++11'de std::clamp yok.
    inline double ClampValue(double value, double low, double high)
    {
        return (value < low) ? low : (value > high) ? high : value;
    }
} // anonim namespace

//==============================================================================
// Metrics
//==============================================================================
TrackingFailureDetector::Metrics::Metrics()
    : psr(0.0), peakValue(0.0), peakRatio(0.0), peakSharpness(0.0),
      responseDistribution(0.0), histSimilarity(0.0), templateSimilarity(0.0),
      ncc(0.0), ssim(0.0),
      nPsr(0.0), nPeakValue(0.0), nPeakRatio(0.0), nPeakSharpness(0.0),
      nResponseDistribution(0.0), nHistSimilarity(0.0),
      nTemplateSimilarity(0.0), nNcc(0.0), nSsim(0.0)
{
}

//==============================================================================
// Kurulum / sifirlama
//==============================================================================
TrackingFailureDetector::TrackingFailureDetector()
{
    Reset();
}

void TrackingFailureDetector::Reset()
{
    state_                 = State::TRACKING;
    initialized_           = false;
    useColorHistogram_     = false;
    rawConfidence_         = INITIAL_CONFIDENCE;
    filteredConfidence_    = INITIAL_CONFIDENCE;
    lowConfidenceCounter_  = 0;
    highConfidenceCounter_ = 0;
    referenceTemplate_.release();
    initialTemplate_.release();
    referenceHistogram_.release();
    lastMetrics_ = Metrics();
}

const char* TrackingFailureDetector::GetStateName() const
{
    switch (state_)
    {
        case State::TRACKING: return "TRACKING";
        case State::SUSPECT:  return "SUSPECT";
        case State::LOST:     return "LOST";
    }
    return "UNKNOWN";
}

//==============================================================================
// Ana giris noktasi
//==============================================================================
bool TrackingFailureDetector::Update(const cv::Mat& frame,
                                     const cv::Rect& roi,
                                     const cv::Mat& responseMap,
                                     double psr)
{
    //--- 0) Girdi dogrulama: gecersiz girdi = dusuk guvenli kare ------------
    if (frame.empty() || responseMap.empty() ||
        responseMap.type() != CV_32FC1 ||
        responseMap.rows < 3 || responseMap.cols < 3)
    {
        ApplyLowConfidenceFrame();
        return state_ != State::LOST;
    }

    cv::Mat patchForHist;
    cv::Mat grayPatch32F;
    if (!ExtractPatches(frame, roi, patchForHist, grayPatch32F))
    {
        ApplyLowConfidenceFrame();
        return state_ != State::LOST;
    }

    //--- Ilk kare: referanslari kur, degerlendirme yapma ----------------------
    if (!initialized_)
    {
        useColorHistogram_ = (frame.channels() == 3);
        InitializeReferences(patchForHist, grayPatch32F);
        initialized_ = true;
        return true;
    }

    //--- 1) Tum metrikleri hesapla ---------------------------------------------
    Metrics m;
    m.psr = psr;

    cv::Point peakLocation;
    m.peakValue            = CalculatePeakValue(responseMap, peakLocation);
    m.peakRatio            = CalculatePeakRatio(responseMap, peakLocation, m.peakValue);
    m.peakSharpness        = CalculatePeakSharpness(responseMap, peakLocation, m.peakValue);
    m.responseDistribution = CalculateResponseDistribution(responseMap);

    m.histSimilarity       = CalculateHistogramSimilarity(patchForHist);
    m.templateSimilarity   = CalculateTemplateSimilarity(grayPatch32F);
    m.ncc                  = CalculateNCC(grayPatch32F);
    m.ssim                 = CalculateSSIM(grayPatch32F, referenceTemplate_);

    //--- 2) Normalize et ----------------------------------------------------------
    NormalizeMetrics(m);

    //--- 3) 0-100 Tracking Confidence Score ---------------------------------------
    rawConfidence_ = CalculateConfidence(m);

    //--- 4) EMA ile zamansal filtreleme --------------------------------------------
    filteredConfidence_ = (1.0 - EMA_ALPHA) * filteredConfidence_
                        +        EMA_ALPHA  * rawConfidence_;

    lastMetrics_ = m;

    //--- 5-8) Durum makinesi ---------------------------------------------------------
    const bool trackingOk = UpdateStateMachine(filteredConfidence_);

    //--- Referans guncelleme: yalnizca saglam takipte, yavas EMA -----------------------
    if (state_ == State::TRACKING &&
        filteredConfidence_ >= REFERENCE_UPDATE_MIN_CONFIDENCE)
    {
        UpdateReferences(patchForHist, grayPatch32F);
    }

    //--- 9) bool cikti: true = "Tracking OK", false = "Tracking Lost" ------------------
    return trackingOk;
}

//==============================================================================
// Response map metrikleri
//==============================================================================

// Ana tepe degeri ve konumu.
double TrackingFailureDetector::CalculatePeakValue(const cv::Mat& responseMap,
                                                   cv::Point& peakLocation)
{
    double minVal = 0.0;
    double maxVal = 0.0;
    cv::minMaxLoc(responseMap, &minVal, &maxVal, nullptr, &peakLocation);
    return maxVal;
}

// 1 - (ikincilTepe / anaTepe). Ana tepe cevresi maskelendikten sonra kalan
// en yuksek deger ikincil tepedir. Benzer nesne / dagilmis response'ta
// ikincil tepe buyur -> metrik duser.
double TrackingFailureDetector::CalculatePeakRatio(const cv::Mat& responseMap,
                                                   const cv::Point& peakLocation,
                                                   double peakValue)
{
    if (peakValue <= NUMERIC_EPSILON)
    {
        return 0.0; // tepe yok -> en kotu durum
    }

    const int shortSide = std::min(responseMap.rows, responseMap.cols);
    const int exclusionRadius =
        std::max(SECOND_PEAK_EXCLUSION_MIN,
                 static_cast<int>(shortSide * SECOND_PEAK_EXCLUSION_RATIO));

    // Ana tepe cevresini haritanin minimumuyla maskele (kopya uzerinde).
    double minVal = 0.0;
    cv::minMaxLoc(responseMap, &minVal, nullptr, nullptr, nullptr);

    cv::Mat masked = responseMap.clone();
    const int x0 = std::max(0, peakLocation.x - exclusionRadius);
    const int y0 = std::max(0, peakLocation.y - exclusionRadius);
    const int x1 = std::min(responseMap.cols - 1, peakLocation.x + exclusionRadius);
    const int y1 = std::min(responseMap.rows - 1, peakLocation.y + exclusionRadius);
    masked(cv::Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1))
        .setTo(static_cast<float>(minVal));

    double secondPeak = 0.0;
    cv::minMaxLoc(masked, nullptr, &secondPeak, nullptr, nullptr);
    secondPeak = std::max(secondPeak, 0.0);

    const double ratio = secondPeak / (peakValue + NUMERIC_EPSILON);
    return ClampValue(1.0 - ratio, 0.0, 1.0);
}

// Tepenin lokal komsuluguna gore ne kadar sivri oldugu.
// (tepe - komsulukOrtalamasi) / tepe  -> keskin tepe ~ 1'e yaklasir,
// blur / duz response'ta 0'a yaklasir.
double TrackingFailureDetector::CalculatePeakSharpness(const cv::Mat& responseMap,
                                                       const cv::Point& peakLocation,
                                                       double peakValue)
{
    if (peakValue <= NUMERIC_EPSILON)
    {
        return 0.0;
    }

    const int radius = SHARPNESS_WINDOW_RADIUS;
    double neighborSum = 0.0;
    int    neighborCount = 0;

    for (int dy = -radius; dy <= radius; ++dy)
    {
        const int y = peakLocation.y + dy;
        if (y < 0 || y >= responseMap.rows) continue;

        const float* row = responseMap.ptr<float>(y);
        for (int dx = -radius; dx <= radius; ++dx)
        {
            const int x = peakLocation.x + dx;
            if (x < 0 || x >= responseMap.cols) continue;
            if (dx == 0 && dy == 0) continue; // tepenin kendisi haric

            neighborSum += static_cast<double>(row[x]);
            ++neighborCount;
        }
    }

    if (neighborCount == 0)
    {
        return 0.0;
    }

    const double neighborMean = neighborSum / neighborCount;
    const double sharpness = (peakValue - neighborMean) / (peakValue + NUMERIC_EPSILON);
    return ClampValue(sharpness, 0.0, 1.0);
}

// APCE (Average Peak-to-Correlation Energy):
//   (Fmax - Fmin)^2 / mean((F - Fmin)^2)
// Tek keskin tepe + dusuk taban -> yuksek APCE.
// Occlusion / dagilmis coklu tepe -> APCE belirgin duser.
double TrackingFailureDetector::CalculateResponseDistribution(const cv::Mat& responseMap)
{
    double minVal = 0.0;
    double maxVal = 0.0;
    cv::minMaxLoc(responseMap, &minVal, &maxVal, nullptr, nullptr);

    cv::Mat shifted;
    responseMap.convertTo(shifted, CV_32F, 1.0, -minVal); // F - Fmin
    cv::Mat squared = shifted.mul(shifted);

    const double meanSquared = cv::mean(squared)[0];
    const double peakSpan    = maxVal - minVal;

    return (peakSpan * peakSpan) / (meanSquared + NUMERIC_EPSILON);
}

//==============================================================================
// Gorunum metrikleri
//==============================================================================

// Referans histograma Bhattacharyya benzerligi: 1 - mesafe, [0,1].
// Renkli girdide HSV H-S 2B histogram (aydinlanma degisimine daha dayanikli),
// gri girdide (IR vb.) tek kanalli histogram.
double TrackingFailureDetector::CalculateHistogramSimilarity(const cv::Mat& patchForHist) const
{
    if (referenceHistogram_.empty())
    {
        return 0.0;
    }

    const cv::Mat currentHist = ComputeHistogram(patchForHist);
    const double bhattacharyya =
        cv::compareHist(referenceHistogram_, currentHist, cv::HISTCMP_BHATTACHARYYA);

    return ClampValue(1.0 - bhattacharyya, 0.0, 1.0);
}

// EMA ile guncellenen referans template'e karsi TM_CCOEFF_NORMED.
// Ayni boyutta iki patch -> 1x1 sonuc. [-1,1] -> [0,1].
// Kisa vadeli gorunum tutarliligini olcer (ani occlusion / yanlis nesne).
double TrackingFailureDetector::CalculateTemplateSimilarity(const cv::Mat& grayPatch32F) const
{
    if (referenceTemplate_.empty())
    {
        return 0.0;
    }

    cv::Mat result;
    cv::matchTemplate(grayPatch32F, referenceTemplate_, result, cv::TM_CCOEFF_NORMED);
    const double score = static_cast<double>(result.at<float>(0, 0));

    return ClampValue((score + 1.0) * 0.5, 0.0, 1.0);
}

// Sifir-ortalamali normalize capraz korelasyon, ILK template'e karsi elle hesap:
//   NCC = sum((a - ma) * (b - mb)) / (||a - ma|| * ||b - mb||)
// Referansi hic guncellenmedigi icin uzun vadeli model DRIFT dedektorudur;
// pozisyon/poz degisiminde dogal olarak dusecegi icin agirligi dusuk tutulur.
double TrackingFailureDetector::CalculateNCC(const cv::Mat& grayPatch32F) const
{
    if (initialTemplate_.empty())
    {
        return 0.0;
    }

    cv::Scalar meanA;
    cv::Scalar stdDevA;
    cv::Scalar meanB;
    cv::Scalar stdDevB;
    cv::meanStdDev(grayPatch32F,    meanA, stdDevA);
    cv::meanStdDev(initialTemplate_, meanB, stdDevB);

    if (stdDevA[0] < NUMERIC_EPSILON || stdDevB[0] < NUMERIC_EPSILON)
    {
        return 0.0; // duz patch -> korelasyon tanimsiz
    }

    cv::Mat zeroMeanA = grayPatch32F     - meanA[0];
    cv::Mat zeroMeanB = initialTemplate_ - meanB[0];

    const double numerator = zeroMeanA.dot(zeroMeanB);
    const double pixelCount = static_cast<double>(grayPatch32F.total());
    const double denominator = stdDevA[0] * stdDevB[0] * pixelCount;

    const double ncc = numerator / (denominator + NUMERIC_EPSILON);
    return ClampValue((ncc + 1.0) * 0.5, 0.0, 1.0); // [-1,1] -> [0,1]
}

// Standart SSIM (Wang et al. 2004), Gauss pencereli, harici kutuphanesiz.
// Her iki girdi de CV_32FC1, [0,255] araliginda ve ayni boyutta olmalidir.
double TrackingFailureDetector::CalculateSSIM(const cv::Mat& img1_32F,
                                              const cv::Mat& img2_32F)
{
    if (img1_32F.empty() || img2_32F.empty() ||
        img1_32F.size() != img2_32F.size())
    {
        return 0.0;
    }

    const cv::Size kernel(SSIM_GAUSS_KERNEL, SSIM_GAUSS_KERNEL);

    cv::Mat mu1;
    cv::Mat mu2;
    cv::GaussianBlur(img1_32F, mu1, kernel, SSIM_GAUSS_SIGMA);
    cv::GaussianBlur(img2_32F, mu2, kernel, SSIM_GAUSS_SIGMA);

    const cv::Mat mu1Sq  = mu1.mul(mu1);
    const cv::Mat mu2Sq  = mu2.mul(mu2);
    const cv::Mat mu1Mu2 = mu1.mul(mu2);

    cv::Mat sigma1Sq;
    cv::Mat sigma2Sq;
    cv::Mat sigma12;
    cv::GaussianBlur(img1_32F.mul(img1_32F), sigma1Sq, kernel, SSIM_GAUSS_SIGMA);
    cv::GaussianBlur(img2_32F.mul(img2_32F), sigma2Sq, kernel, SSIM_GAUSS_SIGMA);
    cv::GaussianBlur(img1_32F.mul(img2_32F), sigma12,  kernel, SSIM_GAUSS_SIGMA);
    sigma1Sq -= mu1Sq;
    sigma2Sq -= mu2Sq;
    sigma12  -= mu1Mu2;

    // SSIM haritasi = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) /
    //                 ((mu1^2 + mu2^2 + C1) * (sigma1^2 + sigma2^2 + C2))
    cv::Mat numerator   = (2.0 * mu1Mu2 + SSIM_C1).mul(2.0 * sigma12 + SSIM_C2);
    cv::Mat denominator = (mu1Sq + mu2Sq + SSIM_C1).mul(sigma1Sq + sigma2Sq + SSIM_C2);

    cv::Mat ssimMap;
    cv::divide(numerator, denominator, ssimMap);

    const double meanSsim = cv::mean(ssimMap)[0];
    return ClampValue(meanSsim, 0.0, 1.0);
}

//==============================================================================
// Karar zinciri
//==============================================================================

double TrackingFailureDetector::NormalizeLinear(double value,
                                                double rangeMin,
                                                double rangeMax)
{
    const double span = rangeMax - rangeMin;
    if (span <= NUMERIC_EPSILON)
    {
        return 0.0;
    }
    return ClampValue((value - rangeMin) / span, 0.0, 1.0);
}

void TrackingFailureDetector::NormalizeMetrics(Metrics& metrics)
{
    metrics.nPsr                  = NormalizeLinear(metrics.psr,
                                                    PSR_NORM_MIN, PSR_NORM_MAX);
    metrics.nPeakValue            = NormalizeLinear(metrics.peakValue,
                                                    PEAK_VALUE_NORM_MIN, PEAK_VALUE_NORM_MAX);
    metrics.nResponseDistribution = NormalizeLinear(metrics.responseDistribution,
                                                    APCE_NORM_MIN, APCE_NORM_MAX);
    metrics.nPeakRatio            = NormalizeLinear(metrics.peakRatio,
                                                    PEAK_RATIO_NORM_MIN, PEAK_RATIO_NORM_MAX);
    metrics.nPeakSharpness        = NormalizeLinear(metrics.peakSharpness,
                                                    SHARPNESS_NORM_MIN, SHARPNESS_NORM_MAX);
    metrics.nHistSimilarity       = NormalizeLinear(metrics.histSimilarity,
                                                    HIST_SIM_NORM_MIN, HIST_SIM_NORM_MAX);
    metrics.nTemplateSimilarity   = NormalizeLinear(metrics.templateSimilarity,
                                                    TEMPLATE_SIM_NORM_MIN, TEMPLATE_SIM_NORM_MAX);
    metrics.nNcc                  = NormalizeLinear(metrics.ncc,
                                                    NCC_NORM_MIN, NCC_NORM_MAX);
    metrics.nSsim                 = NormalizeLinear(metrics.ssim,
                                                    SSIM_NORM_MIN, SSIM_NORM_MAX);
}

double TrackingFailureDetector::CalculateConfidence(const Metrics& metrics)
{
    const double weightedSum =
          WEIGHT_PSR          * metrics.nPsr
        + WEIGHT_PEAK_VALUE   * metrics.nPeakValue
        + WEIGHT_DISTRIBUTION * metrics.nResponseDistribution
        + WEIGHT_PEAK_RATIO   * metrics.nPeakRatio
        + WEIGHT_SHARPNESS    * metrics.nPeakSharpness
        + WEIGHT_HIST_SIM     * metrics.nHistSimilarity
        + WEIGHT_TEMPLATE_SIM * metrics.nTemplateSimilarity
        + WEIGHT_NCC          * metrics.nNcc
        + WEIGHT_SSIM         * metrics.nSsim;

    return ClampValue(weightedSum * CONFIDENCE_MAX, 0.0, CONFIDENCE_MAX);
}

// TRACKING -> SUSPECT -> LOST histerezisli durum makinesi.
// Tek karede asla LOST'a gecilmez: once SUSPECT, ardindan
// LOST_CONSECUTIVE_FRAMES ardisik dusuk kare gerekir.
// Toparlanma da ayni sekilde ardisik yuksek kare ister.
bool TrackingFailureDetector::UpdateStateMachine(double filteredConfidence)
{
    switch (state_)
    {
        case State::TRACKING:
        {
            if (filteredConfidence < CONF_SUSPECT_THRESHOLD)
            {
                state_ = State::SUSPECT;
                lowConfidenceCounter_  = 1;
                highConfidenceCounter_ = 0;
            }
            break;
        }

        case State::SUSPECT:
        {
            if (filteredConfidence < CONF_SUSPECT_THRESHOLD)
            {
                ++lowConfidenceCounter_;
                highConfidenceCounter_ = 0;

                if (lowConfidenceCounter_ >= LOST_CONSECUTIVE_FRAMES)
                {
                    state_ = State::LOST;
                    highConfidenceCounter_ = 0;
                }
            }
            else if (filteredConfidence >= CONF_TRACKING_THRESHOLD)
            {
                ++highConfidenceCounter_;
                lowConfidenceCounter_ = 0;

                if (highConfidenceCounter_ >= RECOVER_FRAMES_FROM_SUSPECT)
                {
                    state_ = State::TRACKING;
                    highConfidenceCounter_ = 0;
                }
            }
            else
            {
                // Ara bolge: iki sayaci da agir agir sifirla, durumda kal.
                highConfidenceCounter_ = 0;
            }
            break;
        }

        case State::LOST:
        {
            if (filteredConfidence >= CONF_TRACKING_THRESHOLD)
            {
                ++highConfidenceCounter_;
                if (highConfidenceCounter_ >= RECOVER_FRAMES_FROM_LOST)
                {
                    state_ = State::TRACKING;
                    highConfidenceCounter_ = 0;
                    lowConfidenceCounter_  = 0;
                }
            }
            else
            {
                highConfidenceCounter_ = 0;
            }
            break;
        }
    }

    return state_ != State::LOST; // true = "Tracking OK", false = "Tracking Lost"
}

//==============================================================================
// Yardimcilar
//==============================================================================

// ROI'yi kare sinirlarina kirpar, iki patch uretir:
//  - patchForHist : orijinal kanal yapisinda, 8U, PATCH_SIZE x PATCH_SIZE
//  - grayPatch32F : gri, CV_32FC1, [0,255], PATCH_SIZE x PATCH_SIZE
bool TrackingFailureDetector::ExtractPatches(const cv::Mat& frame,
                                             const cv::Rect& roi,
                                             cv::Mat& patchForHist,
                                             cv::Mat& grayPatch32F)
{
    const cv::Rect frameRect(0, 0, frame.cols, frame.rows);
    const cv::Rect clipped = roi & frameRect;

    if (clipped.width < MIN_VALID_ROI_SIZE || clipped.height < MIN_VALID_ROI_SIZE)
    {
        return false;
    }

    // Sabit boyuta indir: tum benzerlik metrikleri icin sinirli maliyet.
    cv::Mat resized;
    cv::resize(frame(clipped), resized,
               cv::Size(PATCH_SIZE, PATCH_SIZE), 0.0, 0.0, cv::INTER_LINEAR);

    patchForHist = resized;

    cv::Mat gray8U;
    if (resized.channels() == 3)
    {
        cv::cvtColor(resized, gray8U, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray8U = resized;
    }
    gray8U.convertTo(grayPatch32F, CV_32F); // [0,255] araligi korunur (SSIM sabitleri icin)

    return true;
}

cv::Mat TrackingFailureDetector::ComputeHistogram(const cv::Mat& patchForHist) const
{
    cv::Mat histogram;

    if (useColorHistogram_ && patchForHist.channels() == 3)
    {
        cv::Mat hsv;
        cv::cvtColor(patchForHist, hsv, cv::COLOR_BGR2HSV);

        const int   channels[]  = { 0, 1 };
        const int   histSize[]  = { HIST_HUE_BINS, HIST_SAT_BINS };
        const float hueRange[]  = { 0.0f, HIST_HUE_RANGE_MAX };
        const float satRange[]  = { 0.0f, HIST_SAT_RANGE_MAX };
        const float* ranges[]   = { hueRange, satRange };

        cv::calcHist(&hsv, 1, channels, cv::Mat(), histogram, 2, histSize, ranges);
    }
    else
    {
        cv::Mat gray;
        if (patchForHist.channels() == 3)
        {
            cv::cvtColor(patchForHist, gray, cv::COLOR_BGR2GRAY);
        }
        else
        {
            gray = patchForHist;
        }

        const int   channels[] = { 0 };
        const int   histSize[] = { HIST_GRAY_BINS };
        const float grayRange[] = { 0.0f, HIST_GRAY_RANGE_MAX };
        const float* ranges[]  = { grayRange };

        cv::calcHist(&gray, 1, channels, cv::Mat(), histogram, 1, histSize, ranges);
    }

    cv::normalize(histogram, histogram, 1.0, 0.0, cv::NORM_L1);
    return histogram;
}

void TrackingFailureDetector::InitializeReferences(const cv::Mat& patchForHist,
                                                   const cv::Mat& grayPatch32F)
{
    grayPatch32F.copyTo(referenceTemplate_);
    grayPatch32F.copyTo(initialTemplate_);
    referenceHistogram_ = ComputeHistogram(patchForHist);
}

// Yalnizca TRACKING + yuksek guven durumunda cagrilir.
// Yavas EMA: ani gorunum bozulmalarinin referansa sizmasini engeller,
// yavas poz/aydinlanma degisimlerine adaptasyon saglar.
// initialTemplate_ bilerek hic guncellenmez (drift referansi).
void TrackingFailureDetector::UpdateReferences(const cv::Mat& patchForHist,
                                               const cv::Mat& grayPatch32F)
{
    cv::addWeighted(referenceTemplate_, 1.0 - REFERENCE_EMA_ALPHA,
                    grayPatch32F,             REFERENCE_EMA_ALPHA,
                    0.0, referenceTemplate_);

    const cv::Mat currentHist = ComputeHistogram(patchForHist);
    cv::addWeighted(referenceHistogram_, 1.0 - REFERENCE_EMA_ALPHA,
                    currentHist,                REFERENCE_EMA_ALPHA,
                    0.0, referenceHistogram_);
    cv::normalize(referenceHistogram_, referenceHistogram_, 1.0, 0.0, cv::NORM_L1);
}

// Gecersiz girdi (bos frame, bozuk response map, kare disina tasan ROI)
// sessizce yutulmaz: sifir guvenli bir kare gibi filtreye ve durum
// makinesine islenir. Boylece surekli gecersiz girdi de LOST'a goturur.
void TrackingFailureDetector::ApplyLowConfidenceFrame()
{
    rawConfidence_ = 0.0;
    filteredConfidence_ = (1.0 - EMA_ALPHA) * filteredConfidence_;
    UpdateStateMachine(filteredConfidence_);
}
