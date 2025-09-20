#pragma once
#include <opencv2/core.hpp>
#include <cstdint>

namespace vision {
    cv::Mat yuv420ToRgba(const uint8_t* y, const uint8_t* u, const uint8_t* v,
            int width, int height,
            int yRowStride, int uRowStride, int vRowStride,
            int uPixelStride, int vPixelStride);
} // namespace vision
