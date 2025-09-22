#pragma once
#include <opencv2/core.hpp>

namespace vision {

    void setLastRgba(const cv::Mat& rgba);
    bool getLastRgbaCopy(cv::Mat& out);

}
