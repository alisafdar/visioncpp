#include "visioncpp/tflite_wrapper.hpp"
#include <opencv2/imgproc.hpp>
#include <tensorflow/lite/c/c_api.h>
#include <algorithm>
#include <cassert>
#include <cstring>

namespace vision {

    static TfLiteModel*       g_model  = nullptr;
    static TfLiteInterpreter* g_interpreter = nullptr;
    static std::vector<std::string> g_labels;
    static int  g_inWidth = 300, g_inHeight = 300;
    static bool g_isFloat = false;

    void tfliteSetLabels(const std::vector<std::string>& labels) {
        g_labels = labels;
    }

    bool tfliteInitFromBuffer(const void* model, size_t size, const TfliteOptions& opt) {
        tfliteUnload();

        g_model = TfLiteModelCreate(model, size);
        if (!g_model) {
            return false;
        }

        TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
        TfLiteInterpreterOptionsSetNumThreads(options, opt.numThreads);

        g_interpreter = TfLiteInterpreterCreate(g_model, options);
        TfLiteInterpreterOptionsDelete(options);
        if (!g_interpreter) {
            return false;
        }

        if (TfLiteInterpreterAllocateTensors(g_interpreter) != kTfLiteOk) {
            return false;
        }

        TfLiteTensor* in = TfLiteInterpreterGetInputTensor(g_interpreter, 0);
        if (!in) {
            return false;
        }

        g_isFloat = (TfLiteTensorType(in) == kTfLiteFloat32);

        const int numDims = TfLiteTensorNumDims(in);
        if (numDims >= 3) {
            g_inHeight = TfLiteTensorDim(in, numDims == 4 ? 1 : 0);
            g_inWidth = TfLiteTensorDim(in, numDims == 4 ? 2 : 1);
            if (g_inHeight <= 0 || g_inWidth <= 0) {
                g_inHeight = 300; g_inWidth = 300;
            }
        }
        return true;
    }

    bool tfliteReady() {
        return g_interpreter != nullptr;
    }

    void tfliteUnload() {
        if (g_interpreter) {
            TfLiteInterpreterDelete(g_interpreter);
            g_interpreter = nullptr;
        }

        if (g_model)  {
            TfLiteModelDelete(g_model);
            g_model = nullptr;
        }
    }

    static void feedInput(const cv::Mat& rgba, TfLiteTensor* t, int inW, int inH, bool isFloat) {
        cv::Mat rgb;
        cv::cvtColor(rgba, rgb, cv::COLOR_RGBA2RGB);

        cv::Mat resized;
        cv::resize(rgb, resized, cv::Size(inW, inH), 0,0, cv::INTER_LINEAR);

        void* data = TfLiteTensorData(t);
        if (!data) return;

        if (isFloat) {
            float* destination = static_cast<float*>(data);
            for (int y=0; y<resized.rows; ++y) {
                const uint8_t* row = resized.ptr<uint8_t>(y);
                for (int x=0; x<resized.cols*3; ++x) *destination++ = row[x] / 255.f;
            }
        } else if (TfLiteTensorType(t) == kTfLiteUInt8) {
            uint8_t* destination = static_cast<uint8_t*>(data);
            for (int y=0; y<resized.rows; ++y) {
                const uint8_t* source = resized.ptr<uint8_t>(y);
                std::memcpy(destination + y*resized.cols*3, source, resized.cols*3);
            }
        } else {
            // Unsupported input type — ignore
        }
    }

    std::vector<Detection> tfliteDetect(const cv::Mat& rgba, float scoreThreshold) {
        std::vector<Detection> out;
        if (!tfliteReady()) return out;

        TfLiteTensor* in = TfLiteInterpreterGetInputTensor(g_interpreter, 0);
        if (!in) return out;

        feedInput(rgba, in, g_inWidth, g_inHeight, g_isFloat);

        if (TfLiteInterpreterInvoke(g_interpreter) != kTfLiteOk) return out;

        const TfLiteTensor* t_boxes   = TfLiteInterpreterGetOutputTensor(g_interpreter, 0);
        const TfLiteTensor* t_classes = TfLiteInterpreterGetOutputTensor(g_interpreter, 1);
        const TfLiteTensor* t_scores  = TfLiteInterpreterGetOutputTensor(g_interpreter, 2);
        const TfLiteTensor* t_count   = TfLiteInterpreterGetOutputTensor(g_interpreter, 3);

        auto dataOf = [](const TfLiteTensor* tensor) -> const void* {
            return tensor ? TfLiteTensorData(const_cast<TfLiteTensor*>(tensor)) : nullptr;
        };

        int numbers = 0;
        if (t_count && TfLiteTensorType(t_count) == kTfLiteFloat32) {
            const float* count = static_cast<const float*>(dataOf(t_count));
            if (count) numbers = static_cast<int>(count[0]);
        } else if (t_scores) {
            const int numDims = TfLiteTensorNumDims(t_scores);
            if (numDims >= 2) numbers = TfLiteTensorDim(t_scores, numDims-1);
        }
        numbers = std::max(0, std::min(numbers, 1000));

        const float* boxes   = static_cast<const float*>(dataOf(t_boxes));
        const float* classes = static_cast<const float*>(dataOf(t_classes));
        const float* scores  = static_cast<const float*>(dataOf(t_scores));
        if (!boxes || !classes || !scores) return out;

        const int width = rgba.cols, height = rgba.rows;
        for (int i=0; i<numbers; ++i) {
            const float score = scores[i];
            if (score < scoreThreshold) continue;

            const float ymin = boxes[i*4 + 0];
            const float xmin = boxes[i*4 + 1];
            const float ymax = boxes[i*4 + 2];
            const float xmax = boxes[i*4 + 3];

            Detection detection;
            detection.left   = std::max(0.f, xmin * width);
            detection.top    = std::max(0.f, ymin * height);
            detection.right  = std::min<float>(width-1, xmax * width);
            detection.bottom = std::min<float>(height-1, ymax * height);
            detection.score  = score;
            detection.classId= static_cast<int>(classes[i]);
            out.push_back(detection);
        }
        return out;
    }

} // namespace vision
