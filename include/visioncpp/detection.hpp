#pragma once
#include <vector>
#include <string>
#include <cstddef>
#include <cstdint>

namespace vision {

    struct Detection {
        float left, top, right, bottom;
        float score;
        int classId;
    };

    struct Detections {
        std::vector<float> boxes;   // [N*4]
        std::vector<float> scores;
        std::vector<int>   classes;
        double blur{0}, glarePercent{0}, brightness{0};
        long processingMs{0};
    };

    struct YuvFrame {
        const uint8_t* y; const uint8_t* u; const uint8_t* v;
        int width, height;
        int yRowStride, uRowStride, vRowStride;
        int uPixelStride, vPixelStride;
    };

    struct TfliteOptions {
        bool useGpu{false};
        bool useXnnpack{true};
        int numThreads{2};
    };

    struct InitOptions {
        TfliteOptions tfl;
    };

    struct ProcessConfig {
        double blur{120.0};
        double glarePercent{8.0};
        double brightness{40.0};
        float  score{0.5f};
    };

    bool initialize(const InitOptions& opt);
    Detections processFrame(const YuvFrame& frame, const ProcessConfig& cfg);
    bool encodeFrame(int quality, std::vector<uint8_t>& out);

}
