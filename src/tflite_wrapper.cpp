#include "visioncpp/tflite_wrapper.hpp"
#include <opencv2/imgproc.hpp>
#include <tensorflow/lite/c/c_api.h>
#include <algorithm>
#include <cassert>
#include <cstring>

namespace vision {

    static TfLiteModel*       g_model  = nullptr;
    static TfLiteInterpreter* g_interp = nullptr;
    static std::vector<std::string> g_labels;
    static int  g_inW = 300, g_inH = 300;
    static bool g_isFloat = false;

    void tfliteSetLabels(const std::vector<std::string>& labels) { g_labels = labels; }

    bool tfliteInitFromBuffer(const void* model, size_t size, const TfliteOptions& opt) {
        tfliteUnload();

        g_model = TfLiteModelCreate(model, size);
        if (!g_model) return false;

        TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
        TfLiteInterpreterOptionsSetNumThreads(options, opt.numThreads);
        // (Optional) Add delegates via C API here if needed.

        g_interp = TfLiteInterpreterCreate(g_model, options);
        TfLiteInterpreterOptionsDelete(options);
        if (!g_interp) return false;

        if (TfLiteInterpreterAllocateTensors(g_interp) != kTfLiteOk) return false;

        TfLiteTensor* in = TfLiteInterpreterGetInputTensor(g_interp, 0);
        if (!in) return false;

        g_isFloat = (TfLiteTensorType(in) == kTfLiteFloat32);

        const int nd = TfLiteTensorNumDims(in);
        if (nd >= 3) {
            // Assuming NHWC: [1]=H, [2]=W
            g_inH = TfLiteTensorDim(in, nd == 4 ? 1 : 0);
            g_inW = TfLiteTensorDim(in, nd == 4 ? 2 : 1);
            if (g_inH <= 0 || g_inW <= 0) { g_inH = 300; g_inW = 300; }
        }
        return true;
    }

    bool tfliteReady() { return g_interp != nullptr; }

    void tfliteUnload() {
        if (g_interp) { TfLiteInterpreterDelete(g_interp); g_interp = nullptr; }
        if (g_model)  { TfLiteModelDelete(g_model); g_model = nullptr; }
    }

    static void feedInput(const cv::Mat& rgba, TfLiteTensor* t, int inW, int inH, bool isFloat) {
        cv::Mat rgb;     cv::cvtColor(rgba, rgb, cv::COLOR_RGBA2RGB);
        cv::Mat resized; cv::resize(rgb, resized, cv::Size(inW, inH), 0,0, cv::INTER_LINEAR);

        void* data = TfLiteTensorData(t);
        if (!data) return;

        if (isFloat) {
            float* dst = static_cast<float*>(data);
            for (int y=0; y<resized.rows; ++y) {
                const uint8_t* row = resized.ptr<uint8_t>(y);
                for (int x=0; x<resized.cols*3; ++x) *dst++ = row[x] / 255.f;
            }
        } else if (TfLiteTensorType(t) == kTfLiteUInt8) {
            uint8_t* dst = static_cast<uint8_t*>(data);
            for (int y=0; y<resized.rows; ++y) {
                const uint8_t* src = resized.ptr<uint8_t>(y);
                std::memcpy(dst + y*resized.cols*3, src, resized.cols*3);
            }
        } else {
            // Unsupported input type — ignore
        }
    }

    std::vector<Detection> tfliteDetect(const cv::Mat& rgba, float scoreThreshold) {
        std::vector<Detection> out;
        if (!tfliteReady()) return out;

        TfLiteTensor* in = TfLiteInterpreterGetInputTensor(g_interp, 0);
        if (!in) return out;

        feedInput(rgba, in, g_inW, g_inH, g_isFloat);

        if (TfLiteInterpreterInvoke(g_interp) != kTfLiteOk) return out;

        const TfLiteTensor* t_boxes   = TfLiteInterpreterGetOutputTensor(g_interp, 0);
        const TfLiteTensor* t_classes = TfLiteInterpreterGetOutputTensor(g_interp, 1);
        const TfLiteTensor* t_scores  = TfLiteInterpreterGetOutputTensor(g_interp, 2);
        const TfLiteTensor* t_count   = TfLiteInterpreterGetOutputTensor(g_interp, 3);

        auto dataOf = [](const TfLiteTensor* t) -> const void* {
            return t ? TfLiteTensorData(const_cast<TfLiteTensor*>(t)) : nullptr;
        };

        int N = 0;
        if (t_count && TfLiteTensorType(t_count) == kTfLiteFloat32) {
            const float* cnt = static_cast<const float*>(dataOf(t_count));
            if (cnt) N = static_cast<int>(cnt[0]);
        } else if (t_scores) {
            const int nd = TfLiteTensorNumDims(t_scores);
            if (nd >= 2) N = TfLiteTensorDim(t_scores, nd-1);
        }
        N = std::max(0, std::min(N, 1000));

        const float* boxes   = static_cast<const float*>(dataOf(t_boxes));
        const float* classes = static_cast<const float*>(dataOf(t_classes));
        const float* scores  = static_cast<const float*>(dataOf(t_scores));
        if (!boxes || !classes || !scores) return out;

        const int W = rgba.cols, H = rgba.rows;
        for (int i=0; i<N; ++i) {
            const float s = scores[i];
            if (s < scoreThreshold) continue;

            const float ymin = boxes[i*4 + 0];
            const float xmin = boxes[i*4 + 1];
            const float ymax = boxes[i*4 + 2];
            const float xmax = boxes[i*4 + 3];

            Detection d;
            d.left   = std::max(0.f, xmin * W);
            d.top    = std::max(0.f, ymin * H);
            d.right  = std::min<float>(W-1, xmax * W);
            d.bottom = std::min<float>(H-1, ymax * H);
            d.score  = s;
            d.classId= static_cast<int>(classes[i]);
            out.push_back(d);
        }
        return out;
    }

} // namespace vision
