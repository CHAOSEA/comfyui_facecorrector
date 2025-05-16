# Model Download Guide (模型下载指南)

[English](#model-download-guide) | [简体中文](#模型下载指南)

## Model Download Guide

### Required Models

1. **YOLO Face Detection Model**
   - File: `face_yolov8m.pt`
   - Target Path: `ComfyUI/models/ultralytics/bbox/face_yolov8m.pt`
   - Download Link: [Google Drive](https://drive.google.com/file/d/1-0gw-5F4Ud8LuVfV5mMt0zE4JN0EyIhm/view?usp=sharing)
   - Alternative Link: [Hugging Face](https://huggingface.co/DIAMONIK7777/YOLO/resolve/main/face_yolov8m.pt)

2. **InsightFace Model**
   - File: `buffalo_l` folder
   - Target Path: `ComfyUI/models/insightface/models/buffalo_l`
   - Download Link: [Google Drive](https://drive.google.com/file/d/1-1dZCIzHJNxakNCClRnXW1Vk_JqXqy0p/view?usp=sharing)
   - Alternative Link: [InsightFace Official](https://github.com/deepinsight/insightface/tree/master/model_zoo)

3. **BiSeNet Model**
   - File: `resnet34.onnx`
   - Target Path: `ComfyUI/models/bisenet/resnet34.onnx`
   - Download Link: [Google Drive](https://drive.google.com/file/d/1-2Tpya1-8WqCHO6VhHXBzXPjZtfIYzIf/view?usp=sharing)
   - Alternative Link: [Hugging Face](https://huggingface.co/datasets/DIAMONIK7777/BiSeNet/resolve/main/BiSeNet_resnet34.onnx)

### Manual Download Steps

1. Create the following directories in your ComfyUI installation:
   ```bash
   ComfyUI/models/ultralytics/bbox/
   ComfyUI/models/insightface/models/
   ComfyUI/models/bisenet/
   ```

2. Download each model from the provided links

3. Place the downloaded files in their respective directories:
   - Place `face_yolov8m.pt` in `ComfyUI/models/ultralytics/bbox/`
   - Extract and place `buffalo_l` folder in `ComfyUI/models/insightface/models/`
   - Place `resnet34.onnx` in `ComfyUI/models/bisenet/`

4. Restart ComfyUI

### Troubleshooting

- If you encounter "model not found" errors, verify that the model files are in the correct directories
- Ensure file names match exactly (case-sensitive)
- Check file permissions if models are not being loaded
- For InsightFace model, ensure the entire `buffalo_l` folder structure is preserved

---

## 模型下载指南

### 所需模型

1. **YOLO人脸检测模型**
   - 文件：`face_yolov8m.pt`
   - 目标路径：`ComfyUI/models/ultralytics/bbox/face_yolov8m.pt`
   - 下载链接：[Google Drive](https://drive.google.com/file/d/1-0gw-5F4Ud8LuVfV5mMt0zE4JN0EyIhm/view?usp=sharing)
   - 备用链接：[Hugging Face](https://huggingface.co/DIAMONIK7777/YOLO/resolve/main/face_yolov8m.pt)

2. **InsightFace模型**
   - 文件：`buffalo_l` 文件夹
   - 目标路径：`ComfyUI/models/insightface/models/buffalo_l`
   - 下载链接：[Google Drive](https://drive.google.com/file/d/1-1dZCIzHJNxakNCClRnXW1Vk_JqXqy0p/view?usp=sharing)
   - 备用链接：[InsightFace 官方](https://github.com/deepinsight/insightface/tree/master/model_zoo)

3. **BiSeNet模型**
   - 文件：`resnet34.onnx`
   - 目标路径：`ComfyUI/models/bisenet/resnet34.onnx`
   - 下载链接：[Google Drive](https://drive.google.com/file/d/1-2Tpya1-8WqCHO6VhHXBzXPjZtfIYzIf/view?usp=sharing)
   - 备用链接：[Hugging Face](https://huggingface.co/datasets/DIAMONIK7777/BiSeNet/resolve/main/BiSeNet_resnet34.onnx)

### 手动下载步骤

1. 在ComfyUI安装目录中创建以下文件夹：
   ```bash
   ComfyUI/models/ultralytics/bbox/
   ComfyUI/models/insightface/models/
   ComfyUI/models/bisenet/
   ```

2. 从提供的链接下载各个模型文件

3. 将下载的文件放置到对应目录：
   - 将 `face_yolov8m.pt` 放入 `ComfyUI/models/ultralytics/bbox/`
   - 解压并将 `buffalo_l` 文件夹放入 `ComfyUI/models/insightface/models/`
   - 将 `resnet34.onnx` 放入 `ComfyUI/models/bisenet/`

4. 重启ComfyUI

### 常见问题解决

- 如果遇到"找不到模型"错误，请验证模型文件是否在正确的目录中
- 确保文件名完全匹配（区分大小写）
- 如果模型无法加载，检查文件权限
- 对于InsightFace模型，确保保持完整的 `buffalo_l` 文件夹结构 