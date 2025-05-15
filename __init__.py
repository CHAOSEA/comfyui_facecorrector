import os
from .face_corrector import FaceCorrector
from .face_paster import FacePaster  # 导入新节点
from .face_pose_utils import estimate_face_pose, rotate_image_to_upright  # 导入姿态估计和旋转函数

NODE_CLASS_MAPPINGS = {
    "FaceCorrector": FaceCorrector,
    "FacePaster": FacePaster  # 注册新节点
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceCorrector": "人脸检测与校正 @ CHAOS",
    "FacePaster": "人脸贴回 @ CHAOS"
}

def check_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    comfyui_dir = os.path.dirname(os.path.dirname(current_dir))
    
    # YOLO模型路径
    yolo_model_path = os.path.join(comfyui_dir, "models", "ultralytics", "bbox", "face_yolov8m.pt")
    
    # InsightFace模型路径
    insightface_model_path = os.path.join(comfyui_dir, "models", "insightface", "models", "buffalo_l")
    
    # BiSeNet模型路径
    bisenet_model_path = os.path.join(comfyui_dir, "models", "bisenet", "resnet18.onnx")
    
    print("FaceCorrector节点提示：")
    print("如需使用本节点，请确保以下模型文件存在:")
    print(f"1. YOLO人脸检测模型: {yolo_model_path}")
    print(f"2. InsightFace模型: {insightface_model_path}")
    print(f"3. BiSeNet人脸分割模型: {bisenet_model_path}")

check_models()

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']