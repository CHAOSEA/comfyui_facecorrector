# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-03-20

### Added
- Initial release of ComfyUI FaceCorrector
- Smart face detection with YOLO and InsightFace dual-engine support
- Automatic rotation correction for tilted, inverted, or side-facing faces
- Professional-grade face alignment using arcface algorithm
- High-quality mask generation with jonathandinu and bisenet methods
- Flexible face ratio adjustment through face_ratio parameter
- Stable face detection under various lighting conditions
- Support for extreme angle face processing
- Comprehensive documentation in both Chinese and English

### Changed
- Improved face pose estimation algorithm for more accurate face orientation detection
- Enhanced rotation quality using Lanczos interpolation
- Optimized BiSeNet model loading with fixed resnet34 model
- Unified face_ratio parameter logic between YOLO and arcface modes

### Fixed
- BiSeNet model download issues
- YOLO mode face correction alignment
- Unnecessary image cropping after rotation
- Face detection stability in extreme cases

### Removed
- Redundant bisenet_model_type option for simplified configuration 