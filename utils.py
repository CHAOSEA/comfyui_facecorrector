import os
import numpy as np
import cv2
import torch
import torchvision as tv
from PIL import Image
from skimage import transform as trans
import onnxruntime as ort

# 定义arcface标准关键点位置
arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

# 全局变量，用于缓存BiSeNet模型
BISENET_MODEL = None
BISENET_MODEL_TYPE = None

def get_bisenet_model(model_type=None):
    """获取BiSeNet模型，仅支持resnet34模型
    
    Args:
        model_type: 已废弃的参数，为了向后兼容保留
        
    Returns:
        预加载的ONNX模型
    """
    global BISENET_MODEL
    
    # 如果已经加载了模型，直接返回
    if BISENET_MODEL is not None:
        print(f"使用已加载的BiSeNet模型: resnet34")
        return BISENET_MODEL
    
    # 获取模型目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    comfyui_dir = os.path.dirname(os.path.dirname(current_dir))
    models_dir = os.path.join(comfyui_dir, "models", "bisenet")
    
    # 确保模型目录存在
    os.makedirs(models_dir, exist_ok=True)
    
    # 设置模型路径
    model_path = os.path.join(models_dir, "resnet34.onnx")
    
    # 检查模型文件是否存在
    if os.path.exists(model_path):
        try:
            print(f"正在加载BiSeNet模型: resnet34")
            model = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            BISENET_MODEL = model
            print(f"成功加载BiSeNet模型: resnet34")
            return BISENET_MODEL
        except Exception as e:
            print(f"加载BiSeNet模型失败: {str(e)}")
    
    # 如果模型不存在或加载失败，尝试下载模型
    print("未找到可用的BiSeNet模型，尝试下载模型")
    try:
        download_bisenet_model(models_dir)
        if os.path.exists(model_path):
            model = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            BISENET_MODEL = model
            print(f"成功下载并加载BiSeNet模型")
            return BISENET_MODEL
    except Exception as e:
        print(f"下载或加载BiSeNet模型失败: {str(e)}")
    
    # 如果所有尝试都失败，抛出异常
    raise FileNotFoundError("无法找到或加载BiSeNet模型")

def download_bisenet_model(models_dir):
    """下载BiSeNet模型
    
    Args:
        models_dir: 保存模型的目录
    """
    import urllib.request
    
    # 使用Hugging Face上的可靠模型链接
    url = "https://huggingface.co/datasets/DIAMONIK7777/BiSeNet/resolve/main/BiSeNet_resnet34.onnx"
    target_path = os.path.join(models_dir, "resnet34.onnx")
    
    print(f"正在从 {url} 下载BiSeNet模型...")
    
    # 创建一个临时路径来存放下载的模型
    temp_path = os.path.join(models_dir, "temp.onnx")
    try:
        urllib.request.urlretrieve(url, temp_path)
        # 下载完成后重命名
        os.rename(temp_path, target_path)
        print(f"BiSeNet模型下载完成: {target_path}")
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise Exception(f"下载BiSeNet模型失败: {str(e)}")

def generate_face_mask_bisenet(face_image, model_type=None):
    """使用BiSeNet生成人脸分割蒙版
    
    Args:
        face_image: 输入的人脸图像
        model_type: 已废弃的参数，为了向后兼容保留
        
    Returns:
        人脸分割蒙版
    """
    # 确保图像是正确的格式
    if isinstance(face_image, torch.Tensor):
        face_np = face_image.cpu().numpy()
        if face_np.max() <= 1.0:
            face_np = (face_np * 255).astype(np.uint8)
    else:
        face_np = face_image
    
    # 调整图像大小为模型输入尺寸 (通常是512x512)
    input_size = 512
    face_resized = cv2.resize(face_np, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    
    # 预处理图像
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_input = face_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    face_input = np.expand_dims(face_input, axis=0)
    
    # 获取模型
    try:
        model = get_bisenet_model()
    except Exception as e:
        print(f"获取BiSeNet模型失败: {str(e)}，使用简单蒙版代替")
        # 如果模型加载失败，返回简单的椭圆蒙版
        mask = np.zeros((face_np.shape[0], face_np.shape[1]), dtype=np.float32)
        center = (face_np.shape[1] // 2, face_np.shape[0] // 2)
        axes = (int(face_np.shape[1] * 0.4), int(face_np.shape[0] * 0.5))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (51, 51), 30)
        return mask
    
    # 执行推理
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    result = model.run([output_name], {input_name: face_input})[0]
    
    # 处理输出
    pred = np.argmax(result[0], axis=0)
    
    # 创建蒙版 - 修改这里，包含所有面部特征
    # BiSeNet通常将类别1用于皮肤，类别2-11用于眉毛、眼睛、鼻子、嘴唇等
    mask = np.zeros_like(pred, dtype=np.float32)
    
    # 包含所有面部特征（通常类别1-11都是面部相关的）
    for i in range(1, 12):  # 包含所有面部特征类别
        mask[pred == i] = 1.0
    
    # 特别强调嘴巴区域（通常类别11或12是嘴唇）
    # 确保嘴巴区域被包含在蒙版中
    mouth_classes = [11, 12, 13]  # 可能的嘴巴相关类别
    for mouth_class in mouth_classes:
        mouth_mask = (pred == mouth_class).astype(np.float32)
        if np.sum(mouth_mask) > 0:  # 如果检测到嘴巴
            # 稍微扩大嘴巴区域
            mouth_mask = cv2.dilate(mouth_mask, np.ones((5, 5), np.uint8), iterations=1)
            # 确保嘴巴区域在最终蒙版中
            mask = np.maximum(mask, mouth_mask)
    
    # 平滑蒙版边缘
    mask = cv2.GaussianBlur(mask, (9, 9), 3)
    
    # 调整回原始图像大小
    if face_np.shape[:2] != (input_size, input_size):
        mask = cv2.resize(mask, (face_np.shape[1], face_np.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    return mask

def estimate_norm(lmk, image_size=112, mode='arcface'):
    """估计变换矩阵，将人脸对齐到arcface标准"""
    assert lmk.shape == (5, 2)
    
    # 修改这里，支持任意尺寸
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        # 对于不是112的倍数的尺寸，使用最接近的比例
        ratio = float(image_size) / 112.0
        diff_x = 0
    
    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M

def warp_face_back(image, face_image, warp_matrix, face_mask=None):
    """将处理后的人脸贴回原始图像"""
    # 确保图像是NumPy数组
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image.copy()
    
    if isinstance(face_image, torch.Tensor):
        face_np = face_image.cpu().numpy()
        if face_np.max() <= 1.0:
            face_np = (face_np * 255).astype(np.uint8)
    else:
        face_np = face_image.copy()
    
    # 获取图像尺寸
    h, w = image_np.shape[:2]
    face_h, face_w = face_np.shape[:2]
    
    # 打印调试信息
    print(f"原始图像尺寸: {image_np.shape}")
    print(f"人脸图像尺寸: {face_np.shape}")
    print(f"变换矩阵: \n{warp_matrix}")
    
    # 创建逆变换矩阵
    try:
        # 确保warp_matrix是2x3格式
        if warp_matrix.shape != (2, 3):
            print(f"警告：变换矩阵形状不是(2,3): {warp_matrix.shape}")
            if warp_matrix.shape == (3, 3):
                print("转换3x3矩阵为2x3矩阵")
                warp_matrix = warp_matrix[:2, :]
        
        # 计算逆变换矩阵
        inv_matrix = cv2.invertAffineTransform(warp_matrix)
        print(f"逆变换矩阵: \n{inv_matrix}")
        
        # 验证逆变换矩阵的有效性
        det = inv_matrix[0, 0] * inv_matrix[1, 1] - inv_matrix[0, 1] * inv_matrix[1, 0]
        if abs(det) < 1e-6:
            print(f"警告：逆变换矩阵可能不稳定，行列式={det}")
    except Exception as e:
        print(f"创建逆变换矩阵失败: {e}")
        # 返回原图
        return image_np
    
    # 创建结果图像的副本
    result = image_np.copy()
    
    # 将人脸贴回原始图像
    if face_mask is not None and np.max(face_mask) > 0:
        # 确保蒙版是NumPy数组
        if isinstance(face_mask, torch.Tensor):
            mask_np = face_mask.cpu().numpy()
        else:
            mask_np = face_mask.copy()
        
        print(f"蒙版尺寸: {mask_np.shape}, 最大值: {np.max(mask_np)}, 最小值: {np.min(mask_np)}")
        
        # 确保蒙版与人脸图像尺寸匹配
        if mask_np.shape[:2] != face_np.shape[:2]:
            print(f"调整蒙版尺寸从 {mask_np.shape} 到 {face_np.shape[:2]}")
            # 处理多维蒙版的情况
            if len(mask_np.shape) == 3 and mask_np.shape[0] == 1:
                # 如果是形状为(1, H, W)的蒙版，先调整为(H, W)
                mask_np = mask_np[0]
            
            # 现在进行调整大小操作
            mask_np = cv2.resize(mask_np, (face_w, face_h), interpolation=cv2.INTER_LINEAR)
        
        # 扩展蒙版维度以匹配图像
        if len(mask_np.shape) == 2:
            mask_np = mask_np[:, :, np.newaxis]
        
        # 将蒙版应用到人脸
        face_with_mask = face_np * mask_np
        
        # 将蒙版反向变换到原始图像空间
        try:
            # 计算变换后的图像大小，确保不会裁剪
            corners = np.array([
                [0, 0, 1],
                [face_w-1, 0, 1],
                [face_w-1, face_h-1, 1],
                [0, face_h-1, 1]
            ])
            
            # 应用逆变换矩阵计算变换后的角点坐标
            inv_matrix_3x3 = np.vstack([inv_matrix, [0, 0, 1]])
            transformed_corners = np.dot(corners, inv_matrix_3x3.T)
            
            # 计算变换后的边界
            x_min = max(0, int(np.floor(np.min(transformed_corners[:, 0]))))
            x_max = min(w, int(np.ceil(np.max(transformed_corners[:, 0]))))
            y_min = max(0, int(np.floor(np.min(transformed_corners[:, 1]))))
            y_max = min(h, int(np.ceil(np.max(transformed_corners[:, 1]))))
            
            print(f"变换后的边界: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
            
            # 使用高质量插值方法进行反向变换
            warped_mask = cv2.warpAffine(
                mask_np, 
                inv_matrix, 
                (w, h), 
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            # 确保warped_mask有正确的维度
            if len(warped_mask.shape) == 2:
                warped_mask = warped_mask[:, :, np.newaxis]
            
            # 将人脸反向变换到原始图像空间
            warped_face = cv2.warpAffine(
                face_np, 
                inv_matrix, 
                (w, h), 
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            # 应用蒙版
            # 1. 计算反向蒙版
            inv_mask = 1.0 - warped_mask
            
            # 2. 确保维度匹配
            if len(inv_mask.shape) == 3 and inv_mask.shape[2] == 1 and result.shape[2] == 3:
                inv_mask = np.repeat(inv_mask, 3, axis=2)
            
            # 3. 将原始图像与反向蒙版相乘
            result = result * inv_mask
            
            # 4. 将变换后的人脸添加到结果
            # 首先，将变换后的人脸与蒙版相乘
            warped_face_masked = warped_face * warped_mask
            
            # 然后，将结果添加到图像
            result = result + warped_face_masked
            
            print("成功贴回人脸")
        except Exception as e:
            print(f"贴回人脸失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回原图
            return image_np
    else:
        # 直接将人脸反向变换到原始图像空间
        try:
            warped_face = cv2.warpAffine(
                face_np, 
                inv_matrix, 
                (w, h), 
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            result = warped_face
            print("成功贴回人脸(无蒙版)")
        except Exception as e:
            print(f"贴回人脸失败(无蒙版): {e}")
            import traceback
            traceback.print_exc()
            # 返回原图
            return image_np
    
    # 确保结果在有效范围内
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

def merge_warps(warps):
    """合并多个变换矩阵"""
    if len(warps) == 0:
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    
    result = warps[0]
    for warp in warps[1:]:
        # 将2x3矩阵转换为3x3矩阵
        warp_3x3 = np.vstack([warp, [0, 0, 1]])
        result_3x3 = np.vstack([result, [0, 0, 1]])
        
        # 矩阵乘法
        combined = result_3x3 @ warp_3x3
        
        # 转换回2x3矩阵
        result = combined[:2]
    
    return result

# 添加InsightFace生成面部遮罩的函数
def generate_face_mask_insightface(face_image):
    """使用InsightFace生成人脸分割蒙版"""
    # 导入必要的模块
    from .face_corrector import get_insightface_model
    
    # 确保图像是正确的格式
    if isinstance(face_image, torch.Tensor):
        face_np = face_image.cpu().numpy()
        if face_np.max() <= 1.0:
            face_np = (face_np * 255).astype(np.uint8)
    else:
        face_np = face_image.copy()
    
    # 确保图像是BGR格式（InsightFace需要）
    if face_np.shape[2] == 3:  # RGB
        face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
    else:
        face_bgr = face_np
    
    # 获取InsightFace模型
    model = get_insightface_model()
    
    # 检测人脸
    faces = model.get(face_bgr)
    
    # 如果没有检测到人脸，返回简单的椭圆蒙版
    if len(faces) == 0:
        h, w = face_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        center = (w // 2, h // 2)
        axes = (int(w * 0.4), int(h * 0.5))  # 椭圆长短轴
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (51, 51), 30)
        return mask
    
    # 获取最大的人脸
    max_face = max(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
    
    # 获取关键点
    landmarks = max_face.landmark_2d_106 if hasattr(max_face, 'landmark_2d_106') else max_face.kps
    
    # 创建蒙版
    h, w = face_np.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    
    # 使用关键点创建凸包
    if len(landmarks) > 5:
        # 使用凸包创建面部蒙版
        hull = cv2.convexHull(landmarks.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1.0)
        
        # 稍微扩大面部区域
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    else:
        # 如果关键点太少，使用边界框
        x1, y1, x2, y2 = max_face.bbox.astype(int)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1.0, -1)
    
    # 平滑蒙版边缘
    mask = cv2.GaussianBlur(mask, (31, 31), 11)
    
    return mask

def generate_face_mask(face, aligned_face, aligned_size=512, mask_type="insightface"):
    """生成人脸蒙版"""
    if mask_type == "bisenet":
        # 使用BiSeNet生成精确的人脸蒙版
        return generate_face_mask_bisenet(aligned_face)
    elif mask_type == "insightface":
        # 使用InsightFace生成精确的人脸蒙版
        return generate_face_mask_insightface(aligned_face)
    else:
        # 简单实现：使用椭圆蒙版
        mask = np.zeros((aligned_size, aligned_size), dtype=np.float32)
        center = (aligned_size // 2, aligned_size // 2)
        axes = (int(aligned_size * 0.4), int(aligned_size * 0.5))  # 椭圆长短轴
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        
        # 平滑蒙版边缘
        mask = cv2.GaussianBlur(mask, (51, 51), 30)
        
        return mask

class Face:
    def __init__(self, bbox, landmarks=None, det_score=None):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.landmarks = landmarks  # 存储完整的关键点信息
        self.det_score = det_score
        self.image_idx = 0
        self.img = None
        
        # 计算宽高
        self.x1, self.y1, self.x2, self.y2 = bbox
        self.w = self.x2 - self.x1
        self.h = self.y2 - self.y1
        
        # 存储用于arcface对齐的5个关键点
        self.kps = None
        if landmarks is not None:
            if len(landmarks) >= 5:
                # 如果提供了至少5个关键点，直接使用前5个（眼睛、鼻子、嘴角）
                self.kps = np.array(landmarks[:5], dtype=np.float32)
            elif len(landmarks) == 2:
                # 特殊情况：只有眼睛关键点，尝试合成其他关键点
                left_eye, right_eye = landmarks
                eye_center_x = (left_eye[0] + right_eye[0]) / 2
                eye_center_y = (left_eye[1] + right_eye[1]) / 2
                eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
                
                # 使用眼睛中心下方位置作为鼻尖
                nose_x = eye_center_x
                nose_y = eye_center_y + eye_distance * 0.6
                
                # 使用几何关系估算嘴角位置
                left_mouth_x = eye_center_x - eye_distance * 0.5
                right_mouth_x = eye_center_x + eye_distance * 0.5
                mouth_y = eye_center_y + eye_distance * 1.2
                
                # 合成5点关键点
                self.kps = np.array([
                    left_eye,
                    right_eye,
                    [nose_x, nose_y],
                    [left_mouth_x, mouth_y],
                    [right_mouth_x, mouth_y]
                ], dtype=np.float32)
                
                print("从眼睛关键点合成了完整的5点关键点")
        
    def crop(self, size, factor=1.0, align_mode="arcface"):
        """裁剪人脸区域，支持多种对齐模式
        
        Args:
            size: 输出图像的大小
            factor: 人脸在图像中的高度占比因子 (0.5-1.3)，值越大，人脸在图像中占比越大
            align_mode: 对齐模式 ("arcface")
            
        Returns:
            M: 变换矩阵
            crop: 裁剪后的图像
        """
        if self.img is None:
            raise ValueError("Face.img未设置，无法裁剪")
        
        # 如果没有关键点，使用简单的中心裁剪，但考虑factor因素
        if self.kps is None and self.landmarks is None:
            # 计算中心点
            cx = (self.x1 + self.x2) / 2
            cy = (self.y1 + self.y2) / 2
            
            # 计算裁剪区域大小，考虑factor因素
            # factor越大，裁剪区域越小（人脸占比越大）
            face_size = max(self.w, self.h)
            # 修正：当factor越大，需要缩小裁剪区域，使用除法关系
            crop_size = face_size / factor
            
            # 计算变换矩阵
            M = np.array([
                [size / crop_size, 0, size / 2 - cx * size / crop_size],
                [0, size / crop_size, size / 2 - cy * size / crop_size]
            ], dtype=np.float32)
            
            # 应用变换 - 使用INTER_LANCZOS4提高大尺寸图像的质量
            crop = cv2.warpAffine(self.img.cpu().numpy(), M, (size, size), flags=cv2.INTER_LANCZOS4)
            crop = torch.from_numpy(crop).unsqueeze(0)
            
            return M, crop
        
        # 根据对齐模式选择不同的处理方法
        if align_mode == "arcface":
            print(f"使用arcface模式裁剪人脸，使用factor={factor}")
            
            # 准备5个关键点
            if self.kps is not None and self.kps.shape[0] == 5:
                src_pts = self.kps
            elif self.landmarks is not None and len(self.landmarks) >= 68:
                # 使用dlib 68点模型的关键点
                left_eye = np.mean(self.landmarks[36:42], axis=0)
                right_eye = np.mean(self.landmarks[42:48], axis=0)
                nose = self.landmarks[30]
                left_mouth = self.landmarks[48]
                right_mouth = self.landmarks[54]
                src_pts = np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)
            else:
                # 使用简单的中心裁剪
                cx = (self.x1 + self.x2) / 2
                cy = (self.y1 + self.y2) / 2
                face_size = max(self.w, self.h)
                # 修正：同样，这里使用除法关系
                crop_size = face_size / factor
                
                M = np.array([
                    [size / crop_size, 0, size / 2 - cx * size / crop_size],
                    [0, size / crop_size, size / 2 - cy * size / crop_size]
                ], dtype=np.float32)
                
                print(f"使用简单中心裁剪，face_size={face_size}, crop_size={crop_size}")
                crop = cv2.warpAffine(self.img.cpu().numpy(), M, (size, size), flags=cv2.INTER_LANCZOS4)
                crop = torch.from_numpy(crop).unsqueeze(0)
                return M, crop
            
            # 使用仿照comfyui_facetools的方法，但保持人脸的正脸朝向
            # 首先使用标准arcface变换估计基本变换矩阵
            M = estimate_norm(src_pts, image_size=size)
            
            # 根据factor调整裁剪范围，但不改变对齐方式
            # 修正：应该使用factor，值越大，缩放越小（人脸占比越大）
            S = np.array([
                [factor, 0, 0],
                [0, factor, 0],
                [0, 0, 1]
            ])
            
            # 计算中心点
            cx, cy = size/2, size/2
            
            # 创建以中心点为基准的变换矩阵
            T_center_to_origin = np.array([
                [1, 0, -cx],
                [0, 1, -cy],
                [0, 0, 1]
            ])
            
            T_origin_to_center = np.array([
                [1, 0, cx],
                [0, 1, cy],
                [0, 0, 1]
            ])
            
            # 将M转换为3x3矩阵
            M_3x3 = np.vstack([M, [0, 0, 1]])
            
            # 组合变换：先对齐，然后相对于中心点缩放
            combined_matrix = T_origin_to_center @ S @ T_center_to_origin @ M_3x3
            
            # 提取回2x3矩阵用于cv2.warpAffine
            final_M = combined_matrix[:2, :]
            
            # 应用变换
            crop = cv2.warpAffine(self.img.cpu().numpy(), final_M, (size, size), flags=cv2.INTER_LANCZOS4)
            crop = torch.from_numpy(crop).unsqueeze(0)
            
            print(f"使用arcface模式完成裁剪，factor={factor}")
            return final_M, crop
        
        # 如果指定了其他对齐模式，使用简单的中心裁剪
        print(f"未知的对齐模式: {align_mode}，使用简单的中心裁剪")
        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2
        face_size = max(self.w, self.h)
        # 修正：这里使用除法
        crop_size = face_size / factor
        
        M = np.array([
            [size / crop_size, 0, size / 2 - cx * size / crop_size],
            [0, size / crop_size, size / 2 - cy * size / crop_size]
        ], dtype=np.float32)
        
        crop = cv2.warpAffine(self.img.cpu().numpy(), M, (size, size), flags=cv2.INTER_LANCZOS4)
        crop = torch.from_numpy(crop).unsqueeze(0)
        return M, crop