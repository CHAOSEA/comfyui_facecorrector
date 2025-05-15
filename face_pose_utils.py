import os
import numpy as np
import cv2
import math
import torch
import insightface
from insightface.app import FaceAnalysis

# 全局变量用于缓存 InsightFace 3D 姿态估计模型
FACE_POSE_MODEL = None

def get_face_pose_model():
    """获取用于面部姿态估计的 InsightFace 模型"""
    global FACE_POSE_MODEL
    if FACE_POSE_MODEL is None:
        try:
            # 初始化InsightFace模型，确保使用带3D功能的版本
            current_dir = os.path.dirname(os.path.abspath(__file__))
            comfyui_dir = os.path.dirname(os.path.dirname(current_dir))
            model_root = os.path.join(comfyui_dir, "models", "insightface")
            
            if not os.path.exists(os.path.join(model_root, "models", "buffalo_l")):
                raise FileNotFoundError(f"未找到InsightFace模型文件: {os.path.join(model_root, 'models', 'buffalo_l')}")
            
            # 创建支持3D功能的FaceAnalysis
            FACE_POSE_MODEL = FaceAnalysis(
                name="buffalo_l",
                root=model_root,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                allowed_modules=['detection', 'landmark_3d_68']  # 确保加载3D关键点模块
            )
            FACE_POSE_MODEL.prepare(ctx_id=0, det_size=(640, 640))
            print("InsightFace 3D姿态估计模型加载成功")
        except Exception as e:
            print(f"加载InsightFace 3D姿态估计模型失败: {str(e)}")
            return None
    
    return FACE_POSE_MODEL

def estimate_face_pose(face, image_np):
    """
    估计人脸姿态角度：roll、yaw、pitch
    
    Args:
        face: 人脸对象，包含边界框和关键点信息
        image_np: 原始图像的numpy数组
    
    Returns:
        roll, yaw, pitch: 姿态角度，单位为度
        confidence: 估计的可靠性
    """
    print("开始估计人脸姿态...")
    confidence = 0
    roll, yaw, pitch = 0, 0, 0
    
    try:
        # 尝试使用现有关键点进行姿态估计
        if hasattr(face, 'landmarks') and face.landmarks is not None and len(face.landmarks) >= 5:
            landmarks = face.landmarks
            
            if len(landmarks) >= 68:  # 使用68点关键点进行详细估计
                # 获取眼睛和鼻子的关键点
                left_eye = np.mean(landmarks[36:42], axis=0)
                right_eye = np.mean(landmarks[42:48], axis=0)
                nose = landmarks[33]
                
                # 计算roll角度（基于眼睛连线）
                eye_dy = right_eye[1] - left_eye[1]
                eye_dx = right_eye[0] - left_eye[0]
                roll = np.degrees(np.arctan2(eye_dy, eye_dx))
                
                # 根据鼻子与眼睛中心的角度估计yaw
                eye_center_x = (left_eye[0] + right_eye[0]) / 2
                eye_center_y = (left_eye[1] + right_eye[1]) / 2
                nose_dx = nose[0] - eye_center_x
                nose_dy = nose[1] - eye_center_y
                
                # 鼻子与眼睛中心的距离
                nose_eye_distance = np.sqrt(nose_dx**2 + nose_dy**2)
                
                # 使用鼻子相对于眼睛中心的水平位置估计yaw
                eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
                yaw = np.degrees(np.arcsin(np.clip(nose_dx / (eye_distance/2), -1, 1))) * 0.5
                
                # 使用鼻子相对于眼睛中心的垂直位置估计pitch
                eye_nose_dy = nose[1] - eye_center_y
                pitch = np.degrees(np.arcsin(np.clip(eye_nose_dy / nose_eye_distance, -1, 1))) * 0.5
                
                confidence = 0.8  # 使用68点关键点的置信度较高
            elif len(landmarks) >= 5:  # 使用5点关键点进行简单估计
                # 对于InsightFace的5点关键点：left_eye, right_eye, nose, left_mouth, right_mouth
                left_eye = landmarks[0]
                right_eye = landmarks[1]
                nose = landmarks[2]
                
                # 计算roll角度
                eye_dy = right_eye[1] - left_eye[1]
                eye_dx = right_eye[0] - left_eye[0]
                roll = np.degrees(np.arctan2(eye_dy, eye_dx))
                
                # 简单估计yaw和pitch
                eye_center_x = (left_eye[0] + right_eye[0]) / 2
                face_width = abs(right_eye[0] - left_eye[0])
                yaw = (nose[0] - eye_center_x) / (face_width + 1e-5) * 45  # 粗略估计
                
                # 估计pitch
                eye_center_y = (left_eye[1] + right_eye[1]) / 2
                face_height = abs(nose[1] - eye_center_y)
                pitch = (eye_center_y - nose[1]) / (face_height + 1e-5) * 30  # 粗略估计
                
                confidence = 0.6  # 使用5点关键点的置信度中等
        
        # 如果没有可用的关键点或者置信度低，尝试使用InsightFace进行3D姿态估计
        if confidence < 0.6:
            # 提取人脸区域
            x1, y1, x2, y2 = face.bbox
            x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(image_np.shape[1], int(x2)), min(image_np.shape[0], int(y2))
            face_crop = image_np[y1:y2, x1:x2]
            
            if face_crop.size > 0 and face_crop.shape[0] > 10 and face_crop.shape[1] > 10:
                # 确保为BGR格式（InsightFace需要）
                if len(face_crop.shape) == 3 and face_crop.shape[2] == 3:
                    if isinstance(image_np, np.ndarray) and image_np[0,0,0] < image_np[0,0,2]:  # RGB格式
                        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
                
                # 使用InsightFace进行检测和姿态估计
                model = get_face_pose_model()
                if model is not None:
                    insightface_faces = model.get(face_crop)
                    if insightface_faces and len(insightface_faces) > 0:
                        insightface_face = insightface_faces[0]
                        
                        # 获取InsightFace提供的姿态信息
                        if hasattr(insightface_face, 'pose'):
                            pose = insightface_face.pose
                            if pose is not None and len(pose) >= 3:
                                yaw, pitch, roll = pose
                                confidence = 0.9  # InsightFace 3D姿态估计的置信度很高
                        
                        # 如果没有直接提供姿态信息，但有3D关键点，可以计算姿态
                        elif hasattr(insightface_face, 'landmark_3d_68') and insightface_face.landmark_3d_68 is not None:
                            landmarks_3d = insightface_face.landmark_3d_68
                            
                            # 通过3D关键点计算姿态角度
                            # 这里使用简化的方法，真实项目中可能需要更复杂的3D姿态估计算法
                            left_eye = np.mean(landmarks_3d[36:42], axis=0)
                            right_eye = np.mean(landmarks_3d[42:48], axis=0)
                            nose = landmarks_3d[33]
                            
                            # 计算roll角度
                            eye_dy = right_eye[1] - left_eye[1]
                            eye_dx = right_eye[0] - left_eye[0]
                            roll = np.degrees(np.arctan2(eye_dy, eye_dx))
                            
                            # 简单估计yaw (使用z轴信息)
                            eye_center_z = (left_eye[2] + right_eye[2]) / 2
                            nose_dz = nose[2] - eye_center_z
                            yaw = np.degrees(np.arctan2(nose[0] - (left_eye[0] + right_eye[0])/2, -nose_dz))
                            
                            # 简单估计pitch (使用z轴信息)
                            pitch = np.degrees(np.arctan2(nose[1] - (left_eye[1] + right_eye[1])/2, -nose_dz))
                            
                            confidence = 0.85  # 使用3D关键点计算的置信度较高
    
    except Exception as e:
        print(f"姿态估计过程中出错: {e}")
    
    # 确保角度在合理范围内
    roll = np.clip(roll, -180, 180)
    yaw = np.clip(yaw, -90, 90)
    pitch = np.clip(pitch, -90, 90)
    
    print(f"估计的姿态角度: roll={roll:.1f}°, yaw={yaw:.1f}°, pitch={pitch:.1f}°, 置信度={confidence:.2f}")
    
    return roll, yaw, pitch, confidence

def rotate_image_to_upright(image_np, face, roll, yaw, pitch, confidence):
    """
    旋转图像使人脸朝上，处理各种异常角度
    
    Args:
        image_np: 原始图像的numpy数组
        face: 人脸对象
        roll, yaw, pitch: 姿态角度
        confidence: 姿态估计的可靠性
    
    Returns:
        rotated_image: 旋转后的图像
        rotation_matrix: 旋转变换矩阵
        success: 是否成功旋转
    """
    h, w = image_np.shape[:2]
    
    # 获取人脸中心点
    x1, y1, x2, y2 = face.bbox
    face_center_x = (x1 + x2) / 2
    face_center_y = (y1 + y2) / 2
    center = (face_center_x, face_center_y)
    
    # 确定旋转角度
    rotation_angle = 0
    
    # 如果姿态估计结果不可靠，不进行旋转
    if confidence < 0.5:
        print(f"姿态估计结果不可靠(置信度={confidence:.2f})，不进行旋转")
        # 创建恒等变换矩阵
        rotation_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        return image_np, rotation_matrix, False
    
    # 优先处理roll角度
    if abs(roll) > 10:  # 只处理明显倾斜的情况
        rotation_angle = -roll  # 负号是为了校正倾斜
        
        # 处理极端情况：倒置
        if abs(roll) > 165 or abs(roll) < -165:
            # 如果接近180度倒置，直接旋转180度
            rotation_angle = 180 if roll > 0 else -180
            print(f"检测到面部倒置 (roll={roll:.1f}°)，旋转180度")
        else:
            print(f"检测到面部倾斜 (roll={roll:.1f}°)，旋转{rotation_angle:.1f}度")
    
    # 处理严重的侧脸（基于yaw角度）
    # 注意：这里只处理非常严重的侧脸情况，一般的侧脸由后续的face_ratio参数和alignface处理
    if abs(yaw) > 45 and abs(yaw) < 135:
        # 根据yaw的正负决定旋转方向
        add_angle = 90 if yaw > 0 else -90
        rotation_angle += add_angle
        print(f"检测到严重侧脸 (yaw={yaw:.1f}°)，增加旋转{add_angle}度")
    
    # 如果没有明显需要旋转的角度，直接返回原图
    if abs(rotation_angle) < 5:
        print("无需旋转，保持原始图像")
        # 创建恒等变换矩阵
        rotation_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        return image_np, rotation_matrix, False
    
    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    
    # 改进：确保旋转后不裁剪图像的任何部分
    # 计算原始图像四个角的坐标
    corners = np.array([
        [0, 0],
        [w-1, 0],
        [w-1, h-1],
        [0, h-1]
    ])
    
    # 将角点坐标转换为齐次坐标
    corners_homogeneous = np.hstack([corners, np.ones((4, 1))])
    
    # 应用旋转矩阵计算旋转后的角点坐标
    rotated_corners = np.dot(corners_homogeneous, rotation_matrix.T)
    
    # 计算旋转后图像的边界
    x_min = np.min(rotated_corners[:, 0])
    x_max = np.max(rotated_corners[:, 0])
    y_min = np.min(rotated_corners[:, 1])
    y_max = np.max(rotated_corners[:, 1])
    
    # 计算旋转后图像的新尺寸
    new_w = int(np.ceil(x_max - x_min))
    new_h = int(np.ceil(y_max - y_min))
    
    # 调整旋转矩阵的平移部分，确保所有内容都在新图像中可见
    rotation_matrix[0, 2] -= x_min
    rotation_matrix[1, 2] -= y_min
    
    print(f"旋转角度: {rotation_angle:.1f}度，原始尺寸: {w}x{h}，新尺寸: {new_w}x{new_h}")
    print(f"平移调整: x={-x_min:.1f}, y={-y_min:.1f}")
    
    # 执行旋转操作，使用高质量插值
    rotated_image = cv2.warpAffine(
        image_np, rotation_matrix, (new_w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    print(f"成功旋转图像，角度为{rotation_angle:.1f}度，新尺寸为{new_w}x{new_h}")
    return rotated_image, rotation_matrix, True 