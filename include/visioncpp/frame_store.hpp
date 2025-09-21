#pragma once
#include <opencv2/core.hpp>

namespace vision {

/** Save the latest RGBA frame (deep copy, thread-safe). */
    void setLastRgba(const cv::Mat& rgba);

/** Copy out the latest RGBA frame. Returns false if empty. */
    bool getLastRgbaCopy(cv::Mat& out);

} // namespace vision
