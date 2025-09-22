#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <opencv2/core.hpp>

#include "visioncpp/detection.hpp"
#include "visioncpp/tflite_wrapper.hpp"

namespace vision {

/**
 * Initialize the pipeline with an embedded TFLite model and labels.
 * If 'labels' is empty, uses the embedded labelmap.txt.
 */
    bool initialize(const InitOptions& opt);

/**
 * Preferred entry: converts 3-plane YUV_420_888 -> RGBA, rotates by rotationDeg
 * (0/90/180/270) in native, runs detection, and returns pixel-space boxes in
 * the rotated image coordinates.
 */
    Detections processFrame(const YuvFrame& f,
            const ProcessConfig& cfg,
            int rotationDeg);

/**
 * JPEG-encode the last rotated RGBA frame kept by the pipeline into 'out'.
 * Returns true on success.
 */
    bool encodeFrame(int quality, std::vector<uint8_t>& out);

/**
 * Labels currently used by the pipeline (resolved during initialize).
 */
    const std::vector<std::string>& getLabels();

} // namespace vision
