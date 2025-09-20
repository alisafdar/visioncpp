#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "detection.hpp"

namespace vision {
    bool tfliteInitFromBuffer(const void* model, size_t size, const TfliteOptions&);
    void tfliteUnload();
    bool tfliteReady();
    void tfliteSetLabels(const std::vector<std::string>& labels);
    std::vector<Detection> tfliteDetect(const cv::Mat& rgba, float scoreThreshold);
} // namespace vision
