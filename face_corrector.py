import os
import torch
import numpy as np
import cv2
import torchvision as tv
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from ultralytics import YOLO
from .utils import Face, generate_face_mask, warp_face_back  # 导入 Face 类和贴回函数
from skimage import transform as trans  # 添加这一行导入语句
from .face_pose_utils import estimate_face_pose, rotate_image_to_upright  # 导入姿态估计和旋转函数

# 全局变量，用于缓存模型
INSIGHTFACE_MODEL = None
YOLO_MODEL = None
JONATHANDINU_MODEL = None  # 添加 jonathandinu 模型缓存变量

def get_yolo_model():
    global YOLO_MODEL
    if YOLO_MODEL is None:
        # 使用相对路径获取模型
        current_dir = os.path.dirname(os.path.abspath(__file__))
        comfyui_dir = os.path.dirname(os.path.dirname(current_dir))  # 修改：减少一层目录
        model_path = os.path.join(comfyui_dir, "models", "ultralytics", "bbox", "face_yolov8m.pt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到YOLO模型文件: {model_path}")
        
        YOLO_MODEL = YOLO(model_path)
    return YOLO_MODEL

def get_insightface_model():
    global INSIGHTFACE_MODEL
    if INSIGHTFACE_MODEL is None:
        # 初始化InsightFace模型
        current_dir = os.path.dirname(os.path.abspath(__file__))
        comfyui_dir = os.path.dirname(os.path.dirname(current_dir))  # 修改：减少一层目录
        model_root = os.path.join(comfyui_dir, "models", "insightface")
        
        if not os.path.exists(os.path.join(model_root, "models", "buffalo_l")):
            raise FileNotFoundError(f"未找到InsightFace模型文件: {os.path.join(model_root, 'models', 'buffalo_l')}")
        
        INSIGHTFACE_MODEL = FaceAnalysis(
            name="buffalo_l",
            root=model_root,  # 指定模型位置
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        INSIGHTFACE_MODEL.prepare(ctx_id=0, det_size=(640, 640))
    return INSIGHTFACE_MODEL

def get_jonathandinu_model():
    """获取jonathandinu面部解析模型"""
    global JONATHANDINU_MODEL
    if JONATHANDINU_MODEL is None:
        try:
            # 尝试导入transformers库
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            
            # 创建设备选择逻辑
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
            
            print(f"正在加载jonathandinu面部解析模型到{device}设备...")
            
            # 加载模型和处理器
            image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
            model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
            model.to(device)
            
            # 缓存模型和处理器
            JONATHANDINU_MODEL = {
                "processor": image_processor,
                "model": model,
                "device": device
            }
            
            print("jonathandinu面部解析模型加载成功！")
        except Exception as e:
            print(f"加载jonathandinu面部解析模型失败: {str(e)}")
            print("如需使用此模型，请安装transformers库: pip install transformers")
            # 返回None表示模型加载失败
            return None
    
    return JONATHANDINU_MODEL

def generate_face_mask_jonathandinu(image, included_labels=None):
    """使用jonathandinu面部解析模型生成蒙版
    
    Args:
        image: 输入图像，可以是numpy数组或torch张量
        included_labels: 要包含在蒙版中的标签列表，None表示包含所有面部标签
        
    Returns:
        蒙版，范围为0-1的numpy数组
    """
    # 获取模型
    model_data = get_jonathandinu_model()
    if model_data is None:
        print("jonathandinu模型未加载，使用备用蒙版生成方法")
        # 使用BiSeNet作为备用
        from .utils import generate_face_mask_bisenet
        return generate_face_mask_bisenet(image)
    
    # 提取模型和处理器
    processor = model_data["processor"]
    model = model_data["model"]
    device = model_data["device"]
    
    # 如果未指定标签，使用默认的面部标签（排除背景、帽子、耳环、项链、衣服和头发）
    if included_labels is None:
        included_labels = [
            "skin", "nose", "eye_g", "l_eye", "r_eye", 
            "l_brow", "r_brow", "l_ear", "r_ear", 
            "mouth", "u_lip", "l_lip"
            # 注意："hair"已从列表中移除
        ]
    
    # 标签ID映射
    label_map = {
        "background": 0,
        "skin": 1,
        "nose": 2,
        "eye_g": 3,  # 眼镜
        "l_eye": 4,  # 左眼
        "r_eye": 5,  # 右眼
        "l_brow": 6,  # 左眉毛
        "r_brow": 7,  # 右眉毛
        "l_ear": 8,  # 左耳
        "r_ear": 9,  # 右耳
        "mouth": 10,  # 嘴巴
        "u_lip": 11,  # 上唇
        "l_lip": 12,  # 下唇
        "hair": 13,  # 头发
        "hat": 14,  # 帽子
        "ear_r": 15,  # 耳环
        "neck_l": 16,  # 项链
        "neck": 17,  # 脖子
        "cloth": 18,  # 衣服
    }
    
    try:
        # 转换图像为PIL格式
        if isinstance(image, torch.Tensor):
            # 如果是张量，先转为numpy
            image_np = image.cpu().numpy()
            if len(image_np.shape) == 4:  # 批次维度
                image_np = image_np[0]
                
            # 确保值范围在0-255
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
                
            # 确保通道顺序是RGB
            if image_np.shape[0] == 3:  # CHW格式
                image_np = np.transpose(image_np, (1, 2, 0))
                
            # 转换为PIL图像
            pil_image = Image.fromarray(image_np)
        elif isinstance(image, np.ndarray):
            # 确保值范围在0-255
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
                
            # 确保通道顺序是RGB
            if image.shape[2] == 3 and image[0, 0, 0] > 100 and image[0, 0, 2] < 100:  # 可能是BGR格式
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            # 转换为PIL图像
            pil_image = Image.fromarray(image)
        else:
            # 假设已经是PIL图像
            pil_image = image
        
        # 预处理图像
        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        
        # 进行推理
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 获取预测结果
        logits = outputs.logits
        
        # 上采样回原始图像尺寸
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=pil_image.size[::-1],  # H x W
            mode='bilinear',
            align_corners=False
        )
        
        # 获取每个像素的类别
        pred_labels = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        
        # 创建空白蒙版
        mask = np.zeros_like(pred_labels, dtype=np.float32)
        
        # 将选定的标签添加到蒙版中
        for label in included_labels:
            if label in label_map:
                label_id = label_map[label]
                mask[pred_labels == label_id] = 1.0
        
        # 先用小内核进行形态学操作，填补小洞和移除小噪点
        kernel = np.ones((3, 3), np.uint8)
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        mask = mask_uint8.astype(np.float32) / 255.0
        
        # 增强平滑效果，使用更大的高斯模糊内核和更大的sigma值
        mask = cv2.GaussianBlur(mask, (15, 15), 5)
        
        # 应用额外的边缘平滑处理
        # 查找蒙版边缘
        edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
        
        # 对边缘区域应用更强的平滑
        edge_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
        edge_mask = edge_dilated.astype(np.float32) / 255.0
        
        # 在边缘区域应用额外的高斯模糊
        edge_smoothed = cv2.GaussianBlur(mask, (21, 21), 7)
        
        # 将边缘处的平滑结果与原始蒙版混合
        mask = mask * (1 - edge_mask) + edge_smoothed * edge_mask
        
        # 最终的全局平滑
        mask = cv2.GaussianBlur(mask, (9, 9), 3)
        
        print(f"使用jonathandinu模型成功生成了包含{', '.join(included_labels)}的面部蒙版，已应用增强平滑效果")
        
        return mask
        
    except Exception as e:
        print(f"jonathandinu面部蒙版生成失败: {str(e)}")
        # 使用BiSeNet作为备用
        from .utils import generate_face_mask_bisenet
        return generate_face_mask_bisenet(image)

def detect_faces_yolo(image, threshold=0.5):
    """使用YOLO检测人脸，并增加关键点检测步骤"""
    # 确保图像是正确的格式
    if isinstance(image, torch.Tensor):
        # 转换为uint8类型
        if image.max() <= 1.0:
            image_np = (image * 255).type(torch.uint8).cpu().numpy()
        else:
            image_np = image.type(torch.uint8).cpu().numpy()
    else:
        image_np = image
    
    # 获取YOLO模型
    model = get_yolo_model()
    
    # 检测人脸
    results = model(image_np, conf=threshold)
    
    # 提取检测结果并转换为Face对象
    faces = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            
            # 裁剪人脸区域
            x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
            face_crop = image_np[y1_int:y2_int, x1_int:x2_int]
            
            # 如果裁剪区域有效，则进行关键点检测
            if face_crop.size > 0 and face_crop.shape[0] > 10 and face_crop.shape[1] > 10:
                try:
                    # 尝试使用 InsightFace 进行关键点检测
                    insight_model = get_insightface_model()
                    face_crop_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR) if face_crop.shape[2] == 3 else face_crop
                    insight_faces = insight_model.get(face_crop_bgr)
                    
                    if insight_faces and len(insight_faces) > 0:
                        # 获取第一个检测到的人脸的关键点
                        landmarks = insight_faces[0].kps
                        
                        # 调整关键点坐标到原始图像坐标系
                        landmarks[:, 0] += x1_int
                        landmarks[:, 1] += y1_int
                        
                        face = Face(
                            bbox=[int(x1), int(y1), int(x2), int(y2)],
                            landmarks=landmarks,
                            det_score=confidence
                        )
                        faces.append(face)
                        continue
                except Exception as e:
                    print(f"InsightFace 关键点检测失败: {e}")
            
            # 如果 InsightFace 关键点检测失败，则创建没有关键点的 Face 对象
            face = Face(
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                det_score=confidence
            )
            faces.append(face)
    
    return faces

def detect_faces_insightface(image, threshold=0.5):
    """使用InsightFace检测人脸"""
    # 确保图像是正确的格式
    if isinstance(image, torch.Tensor):
        # 转换为uint8类型
        if image.max() <= 1.0:
            image_np = (image * 255).type(torch.uint8).cpu().numpy()
        else:
            image_np = image.type(torch.uint8).cpu().numpy()
    else:
        image_np = image
    
    # 确保图像是BGR格式（InsightFace需要）
    if image_np.shape[2] == 3:  # RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # 获取InsightFace模型
    model = get_insightface_model()
    
    # 检测人脸
    insightface_faces = model.get(image_np)
    
    # 过滤低置信度的人脸并转换为Face对象
    faces = []
    for face in insightface_faces:
        if face.det_score > threshold:
            x1, y1, x2, y2 = face.bbox
            face_obj = Face(
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                landmarks=face.kps,
                det_score=face.det_score
            )
            faces.append(face_obj)
    
    return faces

def multi_scale_detect(image, method="insightface", threshold=0.5):
    """多尺度检测策略"""
    # 确保图像是uint8类型
    if isinstance(image, torch.Tensor):
        if image.max() <= 1.0:
            image_np = (image * 255).type(torch.uint8).cpu().numpy()
        else:
            image_np = image.type(torch.uint8).cpu().numpy()
    else:
        image_np = image
    
    # 原始尺寸检测
    if method == "yolo":
        try:
            faces = detect_faces_yolo(image_np, threshold)
            if faces and len(faces) > 0:
                print(f"YOLO检测到{len(faces)}个人脸")
                return faces, "yolo"
        except Exception as e:
            print(f"YOLO检测失败: {e}")
    
    if method == "insightface":
        try:
            faces = detect_faces_insightface(image_np, threshold)
            if faces and len(faces) > 0:
                print(f"InsightFace检测到{len(faces)}个人脸")
                return faces, "insightface"
        except Exception as e:
            print(f"InsightFace检测失败: {e}")
    
    # 尝试不同的缩放比例
    scales = [0.5, 0.75, 1.25, 1.5, 2.0]
    for scale in scales:
        try:
            # 调整图像大小
            h, w = image_np.shape[:2]
            resized = cv2.resize(image_np, (int(w * scale), int(h * scale)))
            
            print(f"尝试缩放比例 {scale}，调整后尺寸: {resized.shape}")
            
            # 基于指定的方法进行尝试
            if method == "yolo":
                faces = detect_faces_yolo(resized, threshold)
                if faces and len(faces) > 0:
                    print(f"在缩放比例 {scale} 下YOLO检测到 {len(faces)} 个人脸")
                    # 调整坐标回原始尺寸
                    for face in faces:
                        face.x1 = int(face.x1 / scale)
                        face.y1 = int(face.y1 / scale)
                        face.x2 = int(face.x2 / scale)
                        face.y2 = int(face.y2 / scale)
                        face.bbox = [face.x1, face.y1, face.x2, face.y2]
                        face.w = face.x2 - face.x1
                        face.h = face.y2 - face.y1
                    return faces, "yolo"
            
            if method == "insightface":
                faces = detect_faces_insightface(resized, threshold)
                if faces and len(faces) > 0:
                    print(f"在缩放比例 {scale} 下InsightFace检测到 {len(faces)} 个人脸")
                    # 调整坐标回原始尺寸
                    for face in faces:
                        face.x1 = int(face.x1 / scale)
                        face.y1 = int(face.y1 / scale)
                        face.x2 = int(face.x2 / scale)
                        face.y2 = int(face.y2 / scale)
                        face.bbox = [face.x1, face.y1, face.x2, face.y2]
                        face.w = face.x2 - face.x1
                        face.h = face.y2 - face.y1
                    return faces, "insightface"
        except Exception as e:
            print(f"缩放检测失败 (scale={scale}): {e}")
    
    # 如果所有方法都失败，尝试降低阈值
    if threshold > 0.2:
        print(f"使用较低阈值 {threshold/2} 重新尝试检测")
        return multi_scale_detect(image, method, threshold/2)
    
    return None, None

def align_face(image, face, method="insightface"):
    """对齐人脸，确保眼睛和眉毛水平"""
    if method == "insightface":
        # 使用InsightFace的关键点进行对齐
        landmarks = face.landmark_3d_68 if hasattr(face, 'landmark_3d_68') else face.landmark_2d_106
        
        # 计算左右眼中心点
        left_eye = np.mean(landmarks[36:42], axis=0) if len(landmarks) >= 68 else np.mean(landmarks[60:68], axis=0)
        right_eye = np.mean(landmarks[42:48], axis=0) if len(landmarks) >= 68 else np.mean(landmarks[68:76], axis=0)
        
        # 计算眼睛连线角度
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # 旋转图像使眼睛水平
        center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        h, w = image.shape[:2]
        aligned_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LANCZOS4)
        
        # 更新关键点坐标
        ones = np.ones(shape=(len(landmarks), 1))
        points_ones = np.hstack([landmarks, ones])
        transformed_landmarks = rotation_matrix.dot(points_ones.T).T
        
        return aligned_image, transformed_landmarks, rotation_matrix
    
    elif method == "yolo":
        # 对于YOLO检测的人脸，我们需要先获取关键点
        # 这里简化处理，仅使用边界框进行简单对齐
        x1, y1, x2, y2 = face['bbox']
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # 创建一个恒等变换矩阵（不旋转）
        rotation_matrix = np.array([[1, 0, 0], [0, 1, 0]])
        
        return image, None, rotation_matrix

def smart_crop(image, face, landmarks=None, face_ratio=0.9, method="insightface"):
    """智能裁剪，确保人脸占比符合要求
    
    Args:
        image: 输入图像
        face: 人脸对象
        landmarks: 人脸关键点
        face_ratio: 人脸在图像中的高度占比 (0.5-1.3)，值越大，人脸占比越大
        method: 使用的方法 ("insightface" 或 "yolo")
    
    Returns:
        cropped: 裁剪后的图像
        crop_matrix: 裁剪变换矩阵
    """
    if method == "insightface":
        # 使用InsightFace的关键点进行裁剪
        if landmarks is None:
            landmarks = face.landmark_3d_68 if hasattr(face, 'landmark_3d_68') else face.landmark_2d_106
        
        # 计算人脸区域
        min_x = np.min(landmarks[:, 0])
        min_y = np.min(landmarks[:, 1])
        max_x = np.max(landmarks[:, 0])
        max_y = np.max(landmarks[:, 1])
        
        face_width = max_x - min_x
        face_height = max_y - min_y
        
        # 计算人脸中心点
        face_center_x = (min_x + max_x) / 2
        face_center_y = (min_y + max_y) / 2
        
        # 根据face_ratio计算需要的图像尺寸
        # face_ratio越大，人脸在图像中占比越大
        required_height = face_height / face_ratio
        
        # 计算裁剪区域，确保人脸居中
        crop_size = max(required_height, face_width)  # 使用较大的尺寸确保包含完整人脸
        
        # 计算裁剪区域的中心点（与人脸中心重合）
        crop_center_x = face_center_x
        crop_center_y = face_center_y
        
        # 计算裁剪区域的边界
        crop_left = int(crop_center_x - crop_size / 2)
        crop_right = int(crop_center_x + crop_size / 2)
        crop_top = int(crop_center_y - crop_size / 2)
        crop_bottom = int(crop_center_y + crop_size / 2)
        
        # 确保裁剪区域在图像内
        h, w = image.shape[:2]
        crop_left = max(0, crop_left)
        crop_top = max(0, crop_top)
        crop_right = min(w, crop_right)
        crop_bottom = min(h, crop_bottom)
        
        # 如果裁剪区域超出图像边界，调整中心点以保持人脸居中
        if crop_left == 0:
            crop_right = min(w, int(crop_size))
        elif crop_right == w:
            crop_left = max(0, w - int(crop_size))
            
        if crop_top == 0:
            crop_bottom = min(h, int(crop_size))
        elif crop_bottom == h:
            crop_top = max(0, h - int(crop_size))
        
        # 裁剪图像
        cropped = image[crop_top:crop_bottom, crop_left:crop_right]
        
        # 计算变换矩阵用于后续贴回
        crop_matrix = np.array([
            [1, 0, -crop_left],
            [0, 1, -crop_top],
            [0, 0, 1]
        ])
        
        return cropped, crop_matrix
    
    elif method == "yolo":
        # 对于YOLO检测的人脸，使用边界框进行裁剪
        x1, y1, x2, y2 = face.bbox
        
        face_width = x2 - x1
        face_height = y2 - y1
        
        # 计算人脸中心点
        face_center_x = (x1 + x2) / 2
        face_center_y = (y1 + y2) / 2
        
        # 根据face_ratio计算需要的图像尺寸
        # 修正: 使用与InsightFace相同的计算方式，确保一致性
        # face_ratio越大，人脸占比越大，所以裁剪区域应该越小
        required_height = face_height / face_ratio
        
        # 计算裁剪区域，确保人脸居中
        crop_size = max(required_height, face_width)  # 使用较大的尺寸确保包含完整人脸
        
        # 计算裁剪区域的中心点（与人脸中心重合）
        crop_center_x = face_center_x
        crop_center_y = face_center_y
        
        # 计算裁剪区域的边界
        crop_left = int(crop_center_x - crop_size / 2)
        crop_right = int(crop_center_x + crop_size / 2)
        crop_top = int(crop_center_y - crop_size / 2)
        crop_bottom = int(crop_center_y + crop_size / 2)
        
        # 确保裁剪区域在图像内
        h, w = image.shape[:2]
        crop_left = max(0, crop_left)
        crop_top = max(0, crop_top)
        crop_right = min(w, crop_right)
        crop_bottom = min(h, crop_bottom)
        
        # 如果裁剪区域超出图像边界，调整中心点以保持人脸居中
        if crop_left == 0:
            crop_right = min(w, int(crop_size))
        elif crop_right == w:
            crop_left = max(0, w - int(crop_size))
            
        if crop_top == 0:
            crop_bottom = min(h, int(crop_size))
        elif crop_bottom == h:
            crop_top = max(0, h - int(crop_size))
        
        # 裁剪图像
        cropped = image[crop_top:crop_bottom, crop_left:crop_right]
        
        # 计算变换矩阵用于后续贴回
        crop_matrix = np.array([
            [1, 0, -crop_left],
            [0, 1, -crop_top],
            [0, 0, 1]
        ])
        
        return cropped, crop_matrix

def generate_face_mask(image, face, method="insightface"):
    """生成面部蒙版"""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    
    if method == "insightface":
        # 使用InsightFace的关键点生成蒙版
        landmarks = face.landmark_3d_68 if hasattr(face, 'landmark_3d_68') else face.landmark_2d_106
        
        # 在旋转方法对齐成功后，替换创建蒙版的代码
        # 原代码：
        # mask = torch.ones((size, size), dtype=torch.float32)
        
        # 新代码：生成精确的面部蒙版
        # 创建一个空白蒙版
        face_mask = np.zeros((size, size), dtype=np.float32)
        
        # 将关键点变换到对齐后的坐标系
        transformed_landmarks = cv2.transform(landmarks.reshape(1, -1, 2), rotation_matrix).reshape(-1, 2)
        
        # 创建凸包
        hull = cv2.convexHull(transformed_landmarks.astype(np.int32))
        cv2.fillConvexPoly(face_mask, hull, 1.0)
        
        # 平滑边缘
        face_mask = cv2.GaussianBlur(face_mask, (31, 31), 11)
        
        # 转换为PyTorch张量
        mask = torch.from_numpy(face_mask).float()
        
    elif method == "yolo":
        # 对于YOLO检测的人脸，使用边界框生成简单蒙版
        x1, y1, x2, y2 = face['bbox']
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1.0, -1)
        
        # 平滑边缘
        mask = cv2.GaussianBlur(mask, (51, 51), 30)
    
    return mask

def compute_warp_matrix(rotation_matrix, crop_matrix):
    """计算用于贴回的变换矩阵"""
    # 组合旋转和裁剪矩阵
    if rotation_matrix.shape[0] == 2:  # 2x3矩阵
        rotation_matrix_3x3 = np.vstack([rotation_matrix, [0, 0, 1]])
    else:
        rotation_matrix_3x3 = rotation_matrix
    
    warp_matrix = crop_matrix @ rotation_matrix_3x3
    
    # 返回2x3矩阵用于cv2.warpAffine
    return warp_matrix[:2]

# 修改原来的FaceCorrector类，恢复原有功能
class FaceCorrector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detection_method": (["insightface", "yolo"], {"default": "insightface"}),
                "crop_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "face_ratio": ("FLOAT", {"default": 0.9, "min": 0.5, "max": 1.3, "step": 0.05}),
                "align_mode": (["arcface"], {"default": "arcface"}),
                "mask_type": (["jonathandinu", "bisenet", "none"], {"default": "jonathandinu"}),
                "auto_rotate": (["是", "否"], {"default": "是"})  # 添加自动旋转选项
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "WARP_MATRIX")
    RETURN_NAMES = ("aligned_face", "face_mask", "warp_matrix")
    FUNCTION = "correct_face"
    CATEGORY = "image/face"

    def correct_face(self, image, detection_method="insightface", crop_size=512, face_ratio=0.9, align_mode="arcface", mask_type="jonathandinu", auto_rotate="是"):
        """人脸校正函数
        
        Args:
            image: 输入图像
            detection_method: 人脸检测方法
            crop_size: 输出图像大小
            face_ratio: 人脸在图像中的高度占比 (0.5-1.3)，值越大，人脸占比越大
            align_mode: 对齐模式 (仅支持"arcface")
            mask_type: 蒙版类型 (jonathandinu, bisenet, none)
            auto_rotate: 是否自动旋转调整人脸朝向 (是, 否)
            
        Returns:
            aligned_face: 对齐后的人脸图像
            face_mask: 人脸蒙版
            warp_matrix: 变换矩阵
        """
        # 限制裁剪尺寸上限为2048
        if crop_size > 2048:
            print(f"警告：裁剪尺寸 {crop_size} 超出上限，已自动调整为2048")
            crop_size = 2048
        
        # 输出人脸比例参数
        print(f"=== FaceCorrector参数 ===")
        print(f"使用的检测方法: {detection_method}")
        print(f"裁剪尺寸: {crop_size}")
        print(f"人脸高度占比: {face_ratio}")
        print(f"对齐模式: {align_mode}")
        print(f"蒙版类型: {mask_type}")
        print(f"自动旋转: {auto_rotate}")
        print(f"======================")
        
        # 确保图像是正确的格式
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # 批次维度
                if image.shape[0] > 1:
                    print("警告：只处理批次中的第一张图像")
                image_np = image[0].cpu().numpy()
            else:
                image_np = image.cpu().numpy()
            
            # 确保值范围在0-255
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image
        
        # 检测人脸
        faces, used_method = multi_scale_detect(image_np, detection_method, 0.5)
        
        # 如果没有检测到人脸，返回原始图像而不是空结果
        if not faces or len(faces) == 0:
            print("未检测到人脸，返回原始图像")
            
            # 创建恒等变换矩阵
            empty_warp = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            
            # 确保返回的图像是Tensor格式
            if not isinstance(image, torch.Tensor):
                image_tensor = torch.from_numpy(image_np).float() / 255.0
                if len(image_tensor.shape) == 3:  # 添加批次维度
                    image_tensor = image_tensor.unsqueeze(0)
                
                # 创建与原始图像相同尺寸的全黑蒙版
                h, w = image_np.shape[:2]
                
                # 如果设置了mask_type，尝试使用对应的方法从原图生成蒙版
                if mask_type != "none":
                    try:
                        print(f"尝试使用{mask_type}从原图生成蒙版")
                        if mask_type == "jonathandinu":
                            face_mask = generate_face_mask_jonathandinu(image_np)
                        elif mask_type == "bisenet":
                            from .utils import generate_face_mask_bisenet
                            face_mask = generate_face_mask_bisenet(image_np)
                        
                        # 将蒙版转换为tensor
                        face_mask_tensor = torch.from_numpy(face_mask).unsqueeze(0)
                        print(f"成功从原图生成蒙版，尺寸: {face_mask_tensor.shape}")
                        return image_tensor, face_mask_tensor, empty_warp
                    except Exception as e:
                        print(f"从原图生成蒙版失败: {str(e)}，使用黑色蒙版替代")
                
                # 如果无法生成蒙版或不需要使用特定方法，创建黑色蒙版
                empty_mask = torch.zeros((1, h, w), dtype=torch.float32)
                return image_tensor, empty_mask, empty_warp
            
            # 处理原图已经是tensor格式的情况
            # 创建与原始图像相同尺寸的全黑蒙版
            h, w = image_np.shape[:2]
            
            # 如果设置了mask_type，尝试使用对应的方法从原图生成蒙版
            if mask_type != "none":
                try:
                    print(f"尝试使用{mask_type}从原图生成蒙版")
                    if mask_type == "jonathandinu":
                        face_mask = generate_face_mask_jonathandinu(image_np)
                    elif mask_type == "bisenet":
                        from .utils import generate_face_mask_bisenet
                        face_mask = generate_face_mask_bisenet(image_np)
                    
                    # 将蒙版转换为tensor
                    face_mask_tensor = torch.from_numpy(face_mask).unsqueeze(0)
                    print(f"成功从原图生成蒙版，尺寸: {face_mask_tensor.shape}")
                    return image, face_mask_tensor, empty_warp
                except Exception as e:
                    print(f"从原图生成蒙版失败: {str(e)}，使用黑色蒙版替代")
            
            # 如果无法生成蒙版或不需要使用特定方法，创建黑色蒙版
            empty_mask = torch.zeros((1, h, w), dtype=torch.float32)
            return image, empty_mask, empty_warp
        
        # 获取最大的人脸
        max_face = max(faces, key=lambda face: face.w * face.h)
        
        # ===== 新增：姿态估计和旋转调整 =====
        # 只有在用户选择自动旋转时才执行旋转调整
        rotation_applied = False
        rotation_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        
        if auto_rotate == "是":
            # 估计人脸姿态
            roll, yaw, pitch, pose_confidence = estimate_face_pose(max_face, image_np)
            
            if pose_confidence > 0.5:  # 只有在姿态估计可靠时才考虑旋转
                print(f"人脸姿态估计: roll={roll:.1f}°, yaw={yaw:.1f}°, pitch={pitch:.1f}°")
                
                # 尝试旋转图像使人脸朝上
                rotated_image, rotation_matrix, rotated = rotate_image_to_upright(image_np, max_face, roll, yaw, pitch, pose_confidence)
                
                if rotated:
                    print(f"已旋转图像调整人脸方向")
                    rotation_applied = True
                    
                    # 在旋转后的图像上重新检测人脸
                    rotated_faces, rotated_method = multi_scale_detect(rotated_image, detection_method, 0.5)
                    
                    if rotated_faces and len(rotated_faces) > 0:
                        # 选择旋转后图像中最大的人脸
                        rotated_max_face = max(rotated_faces, key=lambda face: face.w * face.h)
                        
                        # 再次估计姿态，验证旋转效果
                        r_roll, r_yaw, r_pitch, r_confidence = estimate_face_pose(rotated_max_face, rotated_image)
                        
                        # 如果旋转后姿态更好（特别是roll角度更小），采用旋转后的结果
                        if r_confidence > 0.5 and (abs(r_roll) < abs(roll) or abs(r_yaw) < abs(yaw)):
                            print(f"旋转后姿态改善: roll={r_roll:.1f}°, yaw={r_yaw:.1f}°, pitch={r_pitch:.1f}°")
                            image_np = rotated_image
                            max_face = rotated_max_face
                        else:
                            print("旋转未改善人脸姿态，保留原始图像")
                            rotation_applied = False
                    else:
                        print("旋转后未能检测到人脸，保留原始图像")
                        rotation_applied = False
            else:
                print("人脸姿态估计不可靠，跳过旋转调整")
        # ===== 姿态估计和旋转调整结束 =====
        
        # 设置图像
        max_face.img = torch.from_numpy(image_np)
        
        # 裁剪并对齐人脸 - 无论是YOLO还是InsightFace检测的人脸都使用相同的处理方式
        try:
            print(f"开始裁剪人脸，人脸高度占比设置为: {face_ratio}，使用检测方法: {used_method}")
            # 对YOLO和InsightFace模式都使用Face.crop方法进行裁剪和对齐
            M, cropped_face = max_face.crop(crop_size, face_ratio, align_mode)
            print(f"成功使用{align_mode}模式对齐人脸，最终人脸占比: {face_ratio}")
            
            # 如果已经应用了旋转，需要将旋转矩阵与裁剪对齐矩阵合并
            if rotation_applied:
                print("合并旋转和裁剪变换矩阵")
                # 获取旋转前后的图像尺寸
                original_h, original_w = image.shape[1:3] if isinstance(image, torch.Tensor) and len(image.shape) == 4 else image.shape[:2]
                rotated_h, rotated_w = image_np.shape[:2]
                
                # 获取旋转中心点（在旋转前的坐标系中）
                x1, y1, x2, y2 = max_face.bbox
                face_center_x = (x1 + x2) / 2
                face_center_y = (y1 + y2) / 2
                
                print(f"原始图像尺寸: {original_w}x{original_h}, 旋转后尺寸: {rotated_w}x{rotated_h}")
                print(f"人脸中心点: ({face_center_x:.1f}, {face_center_y:.1f})")
                
                # 1. 将旋转矩阵转换为3x3矩阵
                rotation_matrix_3x3 = np.vstack([rotation_matrix, [0, 0, 1]])
                print(f"旋转矩阵 (3x3):\n{rotation_matrix_3x3}")
                
                # 2. 将裁剪矩阵转换为3x3矩阵
                M_3x3 = np.vstack([M, [0, 0, 1]])
                print(f"裁剪矩阵 (3x3):\n{M_3x3}")
                
                # 3. 计算正确的组合变换矩阵
                # 旋转矩阵已经包含了从原始图像到旋转后图像的变换
                # 裁剪矩阵包含了从旋转后图像到最终裁剪图像的变换
                # 正确的组合顺序是：先应用旋转变换，再应用裁剪变换
                combined_M_3x3 = np.matmul(M_3x3, rotation_matrix_3x3)
                print(f"组合矩阵 (3x3):\n{combined_M_3x3}")
                
                # 4. 转回2x3矩阵用于warpAffine
                M = combined_M_3x3[:2, :]
                
                print(f"最终变换矩阵 (2x3):\n{M}")
            
            # 生成蒙版
            if mask_type != "none":
                if mask_type == "jonathandinu":
                    # 使用jonathandinu生成蒙版
                    face_mask = generate_face_mask_jonathandinu(cropped_face[0].numpy())
                elif mask_type == "bisenet":
                    # 使用BiSeNet生成蒙版
                    from .utils import generate_face_mask_bisenet
                    face_mask = generate_face_mask_bisenet(cropped_face[0].numpy())
                else:
                    # 创建全黑蒙版作为默认选项
                    face_mask = np.zeros((crop_size, crop_size), dtype=np.float32)
                
                face_mask_tensor = torch.from_numpy(face_mask).unsqueeze(0)
            else:
                # 如果不需要蒙版，创建全黑蒙版
                face_mask_tensor = torch.zeros((1, crop_size, crop_size), dtype=torch.float32)
            
            # 确保cropped_face是浮点数并且范围在0-1之间
            if cropped_face.max() > 1.0:
                cropped_face = cropped_face.float() / 255.0
            
            print(f"人脸校正完成，输出图像尺寸: {cropped_face.shape}")
            return cropped_face, face_mask_tensor, M
            
        except Exception as e:
            print(f"人脸裁剪失败: {str(e)}")
            # 人脸裁剪失败时也返回原始图像
            
            # 创建恒等变换矩阵
            empty_warp = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            
            # 确保返回的图像是Tensor格式
            if not isinstance(image, torch.Tensor):
                image_tensor = torch.from_numpy(image_np).float() / 255.0
                if len(image_tensor.shape) == 3:  # 添加批次维度
                    image_tensor = image_tensor.unsqueeze(0)
                
                # 创建与原始图像相同尺寸的全黑蒙版
                h, w = image_np.shape[:2]
                empty_mask = torch.zeros((1, h, w), dtype=torch.float32)
                return image_tensor, empty_mask, empty_warp
            
            # 创建与原始图像相同尺寸的全黑蒙版
            h, w = image_np.shape[:2]
            empty_mask = torch.zeros((1, h, w), dtype=torch.float32)
            return image, empty_mask, empty_warp

# 添加新的FacePaster节点类
class FacePaster:
    """人脸贴回节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "corrected_face": ("IMAGE",),
                "warp_matrix": ("WARP_MATRIX",),
                "face_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paste_face"
    CATEGORY = "image/face"

    def paste_face(self, image, corrected_face, warp_matrix, face_mask):
        """将修复后的人脸贴回原图"""
        # 确保图像是正确的格式
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if len(image_np.shape) == 4:  # 批次维度
                image_np = image_np[0]
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image
        
        if isinstance(corrected_face, torch.Tensor):
            face_np = corrected_face.cpu().numpy()
            if len(face_np.shape) == 4:  # 批次维度
                face_np = face_np[0]
            if face_np.max() <= 1.0:
                face_np = (face_np * 255).astype(np.uint8)
        else:
            face_np = corrected_face
        
        if isinstance(face_mask, torch.Tensor):
            mask_np = face_mask.cpu().numpy()
            if len(mask_np.shape) == 4:  # 批次维度
                mask_np = mask_np[0]
        else:
            mask_np = face_mask
        
        # 将修复后的人脸贴回原图
        result = warp_face_back(image_np, face_np, warp_matrix, mask_np)
        
        # 转换回PyTorch张量
        result_tensor = torch.from_numpy(result).float() / 255.0
        if len(result_tensor.shape) == 3:  # 添加批次维度
            result_tensor = result_tensor.unsqueeze(0)
        
        return (result_tensor,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "FaceCorrector": FaceCorrector,
    "FacePaster": FacePaster
}

# 设置显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceCorrector": "人脸检测与矫正 @ CHAOS",
    "FacePaster": "人脸贴回 @ CHAOS"
}