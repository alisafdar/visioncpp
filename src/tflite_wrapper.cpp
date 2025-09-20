#include "vision/tflite_wrapper.hpp"
#include <tensorflow/lite/c/c_api.h>
#include <opencv2/imgproc.hpp>
#include <cassert>
#include <cstring>

namespace vision {

    static TfLiteModel* g_model = nullptr;
    static TfLiteInterpreter* g_interp = nullptr;
    static std::vector<std::string> g_labels;
    static int g_inW=300, g_inH=300;
    static bool g_isFloat = false;

    void tfliteSetLabels(const std::vector<std::string>& labels) { g_labels = labels; }

    bool tfliteInitFromBuffer(const void* model, size_t size, const TfliteOptions& opt) {
        tfliteUnload();
        g_model = TfLiteModelCreate(model, size);
        if (!g_model) return false;

        TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
        TfLiteInterpreterOptionsSetNumThreads(options, opt.numThreads);
        // (Optional) add delegates here if needed (GPU, etc.)

        g_interp = TfLiteInterpreterCreate(g_model, options);
        TfLiteInterpreterOptionsDelete(options);
        if (!g_interp) return false;
        if (TfLiteInterpreterAllocateTensors(g_interp) != kTfLiteOk) return false;

        TfLiteTensor* in = TfLiteInterpreterGetInputTensor(g_interp, 0);
        g_isFloat = (in->type == kTfLiteFloat32);
        if (in->dims->size >= 3) {
            g_inH = in->dims->data[in->dims->size-3];
            g_inW = in->dims->data[in->dims->size-2];
        }
        return true;
    }

    bool tfliteReady(){ return g_interp != nullptr; }

    void tfliteUnload() {
        if (g_interp) { TfLiteInterpreterDelete(g_interp); g_interp=nullptr; }
        if (g_model)  { TfLiteModelDelete(g_model); g_model=nullptr; }
    }

    static void feedInput(const cv::Mat& rgba, TfLiteTensor* t, int inW, int inH, bool isFloat) {
        cv::Mat rgb; cv::cvtColor(rgba, rgb, cv::COLOR_RGBA2RGB);
        cv::Mat resized; cv::resize(rgb, resized, cv::Size(inW, inH), 0,0, cv::INTER_LINEAR);
        if (isFloat) {
            float* dst = reinterpret_cast<float*>(t->data.raw);
            for (int y=0;y<resized.rows;++y) {
                const uint8_t* row = resized.ptr<uint8_t>(y);
                for (int x=0;x<resized.cols*3;++x) *dst++ = row[x] / 255.f;
            }
        } else if (t->type == kTfLiteUInt8) {
            uint8_t* dst = reinterpret_cast<uint8_t*>(t->data.raw);
            for (int y=0;y<resized.rows;++y) {
                const uint8_t* src = resized.ptr<uint8_t>(y);
                memcpy(dst + y*resized.cols*3, src, resized.cols*3);
            }
        } else {
            assert(false && "Unsupported input type");
        }
    }

    std::vector<Detection> tfliteDetect(const cv::Mat& rgba, float scoreThreshold) {
        std::vector<Detection> out;
        if (!tfliteReady()) return out;

        TfLiteTensor* in = TfLiteInterpreterGetInputTensor(g_interp, 0);
        feedInput(rgba, in, g_inW, g_inH, g_isFloat);
        if (TfLiteInterpreterInvoke(g_interp) != kTfLiteOk) return out;

        // SSD-like output arrays:
        const TfLiteTensor* t_boxes   = TfLiteInterpreterGetOutputTensor(g_interp, 0);
        const TfLiteTensor* t_classes = TfLiteInterpreterGetOutputTensor(g_interp, 1);
        const TfLiteTensor* t_scores  = TfLiteInterpreterGetOutputTensor(g_interp, 2);
        const TfLiteTensor* t_count   = TfLiteInterpreterGetOutputTensor(g_interp, 3);

        int N = 0;
        if (t_count && t_count->type == kTfLiteFloat32) {
            N = static_cast<int>(reinterpret_cast<const float*>(t_count->data.raw)[0]);
        } else if (t_scores && t_scores->dims->size>=2) {
            N = t_scores->dims->data[1];
        }
        N = std::max(0, std::min(N, 100));

        const float* boxes = reinterpret_cast<const float*>(t_boxes->data.raw);
        const float* classes = reinterpret_cast<const float*>(t_classes->data.raw);
        const float* scores = reinterpret_cast<const float*>(t_scores->data.raw);

        const int imgW = rgba.cols, imgH = rgba.rows;
        for (int i=0;i<N;++i) {
            float s = scores[i];
            if (s < scoreThreshold) continue;
            const float ymin = boxes[i*4 + 0];
            const float xmin = boxes[i*4 + 1];
            const float ymax = boxes[i*4 + 2];
            const float xmax = boxes[i*4 + 3];

            Detection d;
            d.left   = std::max(0.f, xmin * imgW);
            d.top    = std::max(0.f, ymin * imgH);
            d.right  = std::min<float>(imgW-1, xmax * imgW);
            d.bottom = std::min<float>(imgH-1, ymax * imgH);
            d.score = s;
            d.classId = static_cast<int>(classes[i]);
            out.push_back(d);
        }
        return out;
    }

} // namespace vision
