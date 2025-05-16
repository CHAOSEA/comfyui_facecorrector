import os
import requests
from tqdm import tqdm

def download_bisenet_model():
    """下载BiSeNet模型 (只使用resnet34版本)"""
    # 获取模型目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    comfyui_dir = os.path.dirname(os.path.dirname(current_dir))
    model_dir = os.path.join(comfyui_dir, "models", "bisenet")
    
    # 创建模型目录
    os.makedirs(model_dir, exist_ok=True)
    
    # 设置模型URL和文件路径
    model_url = "https://huggingface.co/datasets/DIAMONIK7777/BiSeNet/resolve/main/BiSeNet_resnet34.onnx"
    model_path = os.path.join(model_dir, "resnet34.onnx")
    
    # 检查模型文件是否已存在
    if os.path.exists(model_path):
        print(f"BiSeNet模型已存在: {model_path}")
        return model_path
    
    # 下载模型
    print(f"正在下载BiSeNet模型: {model_url}")
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        with open(model_path, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        
        progress_bar.close()
        print(f"BiSeNet模型下载完成: {model_path}")
        return model_path
    
    except Exception as e:
        print(f"下载BiSeNet模型失败: {e}")
        raise

# 获取InsightFace模型的函数
def get_insightface_model():
    """获取InsightFace模型"""
    from .face_corrector import get_insightface_model as get_model
    return get_model()