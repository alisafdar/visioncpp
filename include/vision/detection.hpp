#pragma once
#include <vector>
#include <string>
#include <cstddef>
#include <cstdint>

namespace vision {

    struct Detection { float left, top, right, bottom; float score; int classId; };

    struct Detections {
        std::vector<float> boxes;   // [N*4]
        std::vector<float> scores;  // [N]
        std::vector<int>   classes; // [N]
        double blurVar{0}, glarePercent{0}, brightness{0};
        long processingMs{0};
    };

    struct YuvFrame {
        const uint8_t* y; const uint8_t* u; const uint8_t* v;
        int width, height;
        int yRowStride, uRowStride, vRowStride;
        int uPixelStride, vPixelStride;
    };

    struct TfliteOptions { bool useGpu{false}; bool useXnnpack{true}; int numThreads{2}; };
    struct InitOptions { TfliteOptions tfl; };

    struct ProcessConfig {
        double blurThreshold{120.0};
        double glareThresholdPercent{8.0};
        double brightnessFloor{40.0};
        float  scoreThreshold{0.5f};
    };

    bool  pipelineInitializeEmbedded(const std::vector<std::string>& labels, const InitOptions& opt);
    Detections pipelineProcessYuv(const YuvFrame& frame, const ProcessConfig& cfg);

// Encode the last processed RGBA to JPEG. Returns true and fills 'out' if available.
    bool pipelineEncodeLastRgbaToJpeg(int quality, std::vector<uint8_t>& out);

} // namespace vision
