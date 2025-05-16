# ComfyUI FaceCorrector

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/CHAOSEA/comfyui_facecorrector?style=flat-square)](https://github.com/CHAOSEA/comfyui_facecorrector/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/CHAOSEA/comfyui_facecorrector?style=flat-square)](https://github.com/CHAOSEA/comfyui_facecorrector/network/members)
[![GitHub issues](https://img.shields.io/github/issues/CHAOSEA/comfyui_facecorrector?style=flat-square)](https://github.com/CHAOSEA/comfyui_facecorrector/issues)
[![GitHub license](https://img.shields.io/github/license/CHAOSEA/comfyui_facecorrector?style=flat-square)](https://github.com/CHAOSEA/comfyui_facecorrector/blob/main/LICENSE)

English | [简体中文](README.md)

A ComfyUI node for face processing, providing face detection, correction, alignment, and mask generation capabilities.

</div>

## 💡 Main Applications

This node is primarily designed for:
- Image preprocessing before face/head swapping
- Preprocessing before facial detail refinement (eyes, eyebrows, facial features)
- Scenarios requiring unified face size and orientation

**Key Advantages**:
- No manual adjustment of input image size and angle required
- No repeated adjustment of detection parameters and thresholds needed
- One-click generation of corrected faces with specified size and ratio

## 📸 Workflow Examples

### InsightFace Mode
![InsightFace Mode Workflow](workflow/facecorrector_workflow_insightface.png)

### YOLO Mode
![YOLO Mode Workflow](workflow/facecorrector_workflow_yolo.png)

## ✨ Features

- 🎯 **Smart Face Detection**: Integrated YOLO and InsightFace dual-engine detection
- 🔄 **Automatic Correction**: Supports processing faces in various scenarios:
  - Inverted faces (e.g., from upside-down photos)
  - Tilted faces (over 10 degrees from horizontal)
  - Side-view faces (large-angle profile shots)
  - Faces with partial occlusion or closed eyes
- ⚖️ **Precise Alignment**: Professional-grade face alignment using arcface algorithm
- 🎭 **Mask Generation**: Multiple mask generation methods (jonathandinu, bisenet)
- 📏 **Ratio Control**: Precise control of face proportion in output images

## 🚀 Quick Start

### Requirements

- Latest version of ComfyUI
- Python 3.8+
- CUDA support (recommended)

### Installation

1. Clone the repository to your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/CHAOSEA/comfyui_facecorrector.git
```

2. Install dependencies:
```bash
cd comfyui_facecorrector
pip install -r requirements.txt
```

3. Restart ComfyUI

### First Use Notes

- Required models will be downloaded automatically on first run
- Model files will be saved in the corresponding `ComfyUI/models/` directory
- Ensure internet connectivity for model downloads

### Required Models

This node requires the following model files:

1. YOLO Face Detection Model:
   - Path: `ComfyUI/models/ultralytics/bbox/face_yolov8m.pt`
   - Purpose: Fast and accurate face detection

2. InsightFace Model:
   - Path: `ComfyUI/models/insightface/models/buffalo_l`
   - Purpose: Face analysis and landmark detection

3. BiSeNet Model:
   - Path: `ComfyUI/models/bisenet/resnet34.onnx`
   - Purpose: Precise face segmentation mask generation

These models will be downloaded automatically on first run. If download fails, please refer to the [Model Download Guide](docs/model_download_guide.md) for manual download.

## 📖 Usage Guide

### FaceCorrector Node

- **Input Parameters**:
  - `image`: Image to process
  - `detection_method`: Face detection method (insightface, yolo)
  - `crop_size`: Output image size
  - `face_ratio`: Face height ratio in image (0.5-1.3)
  - `align_mode`: Alignment mode (arcface)
  - `mask_type`: Mask type (jonathandinu, bisenet, none)


- **Outputs**:
  - `aligned_face`: Aligned face image
  - `face_mask`: Face mask
  - `warp_matrix`: Transformation matrix

### FacePaster Node

- **Input Parameters**:
  - `image`: Original image
  - `corrected_face`: Modified face image
  - `face_mask`: Face mask for smooth blending
  - `use_mask`: Use mask (yes, no)
  - `warp_matrix`: (Optional) Transformation matrix from FaceCorrector

- **Output**:
  - `image`: Final composited image

## 🔧 Advanced Configuration

### face_ratio Parameter

The `face_ratio` parameter controls face proportion in the output image, ranging from 0.5 to 1.3:
- Higher values increase face proportion (more focus on facial features)
- Lower values decrease face proportion (includes more background)

## 📋 FAQ

1. **Q: What if model download fails?**
   A: You can manually download model files and place them in the corresponding directory. See [Model Download Guide](docs/model_download_guide.md)

2. **Q: Is batch processing supported?**
   A: Currently supports batch input, but processes internally one by one.

## 🤝 Contributing

Contributions and suggestions are welcome! Please check our [Contributing Guide](CONTRIBUTING.md).

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed update history.

## ⭐ Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) - High-quality face detection and analysis
- [YOLOv8](https://github.com/ultralytics/ultralytics) - Fast and accurate object detection
- [BiSeNet](https://github.com/CoinCheung/BiSeNet) - Precise face parsing
- All contributors and users for valuable feedback

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 🌟 Citation

If you use this project in your research, please cite:

```bibtex
@software{comfyui_facecorrector,
  author = {CHAOS},
  title = {ComfyUI FaceCorrector},
  year = {2025},
  url = {https://github.com/CHAOSEA/comfyui_facecorrector}
}
```

## 🌟 Support & Feedback

If this node has been helpful in your workflow, please consider giving us a Star ⭐. Your support motivates us to keep improving!

### Community Support

- Having technical issues?
- Need customization?
- Want to share experiences?

Join our QQ community group: 247228975

We are committed to providing professional technical support and solutions for every user. 