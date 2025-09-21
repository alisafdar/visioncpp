#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <opencv2/core.hpp>

#include "visioncpp/detection.hpp"       // YuvFrame, ProcessConfig, Detections
#include "visioncpp/tflite_wrapper.hpp"  // InitOptions

namespace vision {

/**
 * Initialize the pipeline with an embedded TFLite model and labels.
 * If 'labels' is empty, uses the embedded labelmap.txt.
 */
    bool pipelineInitializeEmbedded(const std::vector<std::string>& labels,
            const InitOptions& opt);

/**
 * Preferred entry: converts 3-plane YUV_420_888 -> RGBA, rotates by rotationDeg
 * (0/90/180/270) in native, runs detection, and returns pixel-space boxes in
 * the rotated image coordinates.
 */
    Detections pipelineProcessYuvRotated(const YuvFrame& f,
            const ProcessConfig& cfg,
            int rotationDeg);

/**
 * Backward-compatible entry: no rotation (equivalent to rotationDeg = 0).
 */
    Detections pipelineProcessYuv(const YuvFrame& f,
            const ProcessConfig& cfg);

/**
 * JPEG-encode the last rotated RGBA frame kept by the pipeline into 'out'.
 * Returns true on success.
 */
    bool pipelineEncodeLastRgbaToJpeg(int quality, std::vector<uint8_t>& out);

/**
 * Labels currently used by the pipeline (resolved during initialize).
 */
    const std::vector<std::string>& pipelineLabels();

} // namespace vision
