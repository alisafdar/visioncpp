#include "visioncpp/detection.hpp"
#include "visioncpp/pipeline.hpp"
#include "visioncpp/tflite_wrapper.hpp"
#include "visioncpp/frame_store.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <mutex>
#include <vector>
#include <cstring>
#include <algorithm>


extern "C" {
extern unsigned char detect[];
extern unsigned int  detect_len;

extern unsigned char labelmap[];
extern unsigned int  labelmap_len;
}

namespace vision {

    static std::vector<std::string> g_labels;

    static double laplacianVar(const cv::Mat& gray) {
        cv::Mat lap; cv::Laplacian(gray, lap, CV_64F);
        cv::Scalar mu, sigma; cv::meanStdDev(lap, mu, sigma);
        return sigma.val[0] * sigma.val[0];
    }
    static double glarePercent(const cv::Mat& gray) {
        int bright = cv::countNonZero(gray > 250);
        return 100.0 * static_cast<double>(bright) / static_cast<double>(gray.total());
    }
    static double meanBrightness(const cv::Mat& gray) {
        return cv::mean(gray)[0];
    }

    static std::vector<std::string> splitLines(const char* data, size_t n) {
        std::vector<std::string> out; out.reserve(128);
        size_t i=0, start=0;
        while (i<n) {
            if (data[i]=='\n' || i==n-1) {
                size_t end = (data[i]=='\n') ? i : i+1;
                std::string s(data+start, data+end);
                if (!s.empty() && s.back()=='\r') s.pop_back();
                if (!s.empty()) out.push_back(s);
                start = i+1;
            }
            ++i;
        }
        return out;
    }

    static cv::Mat yuv420ToRgba(const uint8_t* y, const uint8_t* u, const uint8_t* v,
            int width, int height,
            int yRowStride, int uRowStride, int vRowStride,
            int uPixelStride, int vPixelStride) {
        const int ySize = width * height;
        const int cW = width / 2, cH = height / 2, cSize = cW * cH;

        std::vector<uint8_t> i420;
        i420.resize(ySize + 2 * cSize);

        for (int r = 0; r < height; ++r) {
            std::memcpy(i420.data() + r * width, y + r * yRowStride, width);
        }
        uint8_t* uDst = i420.data() + ySize;
        uint8_t* vDst = uDst + cSize;
        for (int r = 0; r < cH; ++r) {
            const uint8_t* uRow = u + r * uRowStride;
            const uint8_t* vRow = v + r * vRowStride;
            for (int c = 0; c < cW; ++c) {
                uDst[r * cW + c] = uRow[c * uPixelStride];
                vDst[r * cW + c] = vRow[c * vPixelStride];
            }
        }

        cv::Mat src(height + height / 2, width, CV_8UC1, i420.data());
        cv::Mat rgba(height, width, CV_8UC4);
        cv::cvtColor(src, rgba, cv::COLOR_YUV2RGBA_I420);
        return rgba;
    }

    static inline void rotateRgbaInPlace(cv::Mat& img, int rotationDeg) {
        const int r = ((rotationDeg % 360) + 360) % 360;
        if (r == 90) {
            cv::transpose(img, img);
            cv::flip(img, img, 1);
        } else if (r == 270) {
            cv::transpose(img, img);
            cv::flip(img, img, 0);
        } else if (r == 180) {
            cv::flip(img, img, -1);
        }
    }

    bool pipelineInitializeEmbedded(const std::vector<std::string>& labels, const InitOptions& opt) {
        std::vector<std::string> lbs = labels;
        if (lbs.empty()) {
            lbs = splitLines(reinterpret_cast<const char*>(labelmap), static_cast<size_t>(labelmap_len));
        }
        g_labels = lbs;
        tfliteSetLabels(g_labels);
        return tfliteInitFromBuffer(detect,
                static_cast<size_t>(detect_len),
                opt.tfl);
    }

    Detections pipelineProcessYuvRotated(const YuvFrame& f, const ProcessConfig& cfg, int rotationDeg) {
        auto t0 = std::chrono::steady_clock::now();
        Detections out;

        cv::Mat rgba = yuv420ToRgba(f.y, f.u, f.v, f.width, f.height,
                f.yRowStride, f.uRowStride, f.vRowStride,
                f.uPixelStride, f.vPixelStride);

        rotateRgbaInPlace(rgba, rotationDeg);

        setLastRgba(rgba);

        cv::Mat gray; cv::cvtColor(rgba, gray, cv::COLOR_RGBA2GRAY);
        out.blurVar      = laplacianVar(gray);
        out.glarePercent = glarePercent(gray);
        out.brightness   = meanBrightness(gray);

        const auto det = tfliteDetect(rgba, cfg.scoreThreshold);
        out.boxes.reserve(det.size() * 4);
        out.scores.reserve(det.size());
        out.classes.reserve(det.size());

        for (const auto& d : det) {
            out.boxes.push_back(d.left);
            out.boxes.push_back(d.top);
            out.boxes.push_back(d.right);
            out.boxes.push_back(d.bottom);
            out.scores.push_back(d.score);
            out.classes.push_back(d.classId);
        }

        auto t1 = std::chrono::steady_clock::now();
        out.processingMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        return out;
    }

    Detections pipelineProcessYuv(const YuvFrame& f, const ProcessConfig& cfg) {
        return pipelineProcessYuvRotated(f, cfg, 0);
    }

    bool pipelineEncodeLastRgbaToJpeg(int quality, std::vector<uint8_t>& out) {
        cv::Mat rgba;
        if (!getLastRgbaCopy(rgba)) return false;
        cv::Mat bgr;
        cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
        std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, std::max(0, std::min(100, quality)) };
        return cv::imencode(".jpg", bgr, out, params);
    }

    const std::vector<std::string>& pipelineLabels() { return g_labels; }

} // namespace vision
