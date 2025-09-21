//
// Created by Ali Safdar Hayat Khan on 21.09.25.
//
#include "visioncpp/frame_store.hpp"
#include <mutex>

namespace {
    cv::Mat s_lastRgba;
    std::mutex s_mtx;
}

namespace vision {

    void setLastRgba(const cv::Mat& rgba) {
        std::lock_guard<std::mutex> lock(s_mtx);
        rgba.copyTo(s_lastRgba);
    }

    bool getLastRgbaCopy(cv::Mat& out) {
        std::lock_guard<std::mutex> lock(s_mtx);
        if (s_lastRgba.empty()) return false;
        out = s_lastRgba.clone();
        return true;
    }

} // namespace vision
