import os
import numpy as np
import cv2
import math
import torch
import mediapipe as mp
from .utils import Face

# 全局变量用于缓存 MediaPipe 人脸网格检测器
MP_FACE_MESH = None

def get_mediapipe_face_mesh():
    """获取 MediaPipe 人脸网格检测器"""
    global MP_FACE_MESH
    if MP_FACE_MESH is None:
        try:
            # 创建人脸网格检测器
            MP_FACE_MESH = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,  # 静态图像模式
                max_num_faces=1,         # 默认只检测一张脸
                refine_landmarks=True,   # 更精细的关键点
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("MediaPipe 人脸网格检测器加载成功")
        except Exception as e:
            print(f"加载 MediaPipe 人脸网格检测器失败: {str(e)}")
            return None
    
    return MP_FACE_MESH

def detect_face_with_mediapipe(image_np):
    """
    使用 MediaPipe 检测人脸并提取关键点
    
    Args:
        image_np: 输入图像的 numpy 数组，RGB 格式
    
    Returns:
        faces: 检测到的人脸列表，每个人脸包含边界框和关键点信息
        landmarks_list: 完整的 MediaPipe 关键点列表
    """
    # 确保图像是 RGB 格式
    if image_np.shape[2] == 3 and image_np[0,0,0] > image_np[0,0,2]:  # 如果是 BGR 格式
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image_np
    
    # 获取图像尺寸
    height, width = image_rgb.shape[:2]
    
    # 获取 MediaPipe 人脸网格检测器
    face_mesh = get_mediapipe_face_mesh()
    if face_mesh is None:
        print("未能加载 MediaPipe 人脸网格检测器")
        return [], []
    
    # 处理图像
    results = face_mesh.process(image_rgb)
    
    faces = []
    landmarks_list = []
    
    # 检查是否检测到人脸
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 将关键点从相对坐标转换为绝对像素坐标
            landmarks = np.array([
                [int(landmark.x * width), int(landmark.y * height), landmark.z]
                for landmark in face_landmarks.landmark
            ])
            
            # 存储完整的关键点列表
            landmarks_list.append(landmarks)
            
            # 计算边界框
            x_min = int(np.min(landmarks[:, 0]))
            y_min = int(np.min(landmarks[:, 1]))
            x_max = int(np.max(landmarks[:, 0]))
            y_max = int(np.max(landmarks[:, 1]))
            
            # 扩大边界框以包含整个人脸
            bbox_expansion = 0.1  # 扩展原始边界框的10%
            x_expand = int((x_max - x_min) * bbox_expansion)
            y_expand = int((y_max - y_min) * bbox_expansion)
            
            x_min = max(0, x_min - x_expand)
            y_min = max(0, y_min - y_expand)
            x_max = min(width, x_max + x_expand)
            y_max = min(height, y_max + y_expand)
            
            # 创建 5 点关键点 (左眼, 右眼, 鼻尖, 左嘴角, 右嘴角)
            # MediaPipe 关键点索引：
            # - 左眼：133 (左眼中心)
            # - 右眼：362 (右眼中心)
            # - 鼻尖：4
            # - 左嘴角：61
            # - 右嘴角：291
            five_points = np.array([
                landmarks[133, :2],  # 左眼
                landmarks[362, :2],  # 右眼
                landmarks[4, :2],    # 鼻尖
                landmarks[61, :2],   # 左嘴角
                landmarks[291, :2]   # 右嘴角
            ])
            
            # 创建 Face 对象
            face = Face(
                bbox=[x_min, y_min, x_max, y_max],
                landmarks=five_points,  # 使用 5 点关键点
                det_score=0.9  # MediaPipe 一般有较高的置信度
            )
            
            # 存储额外的信息
            face.mediapipe_landmarks = landmarks
            
            faces.append(face)
    
    return faces, landmarks_list

def calculate_roll_angle(landmarks, method='robust'):
    """
    计算人脸的 roll 角度（倾斜角度）
    
    Args:
        landmarks: MediaPipe 的完整人脸关键点
        method: 使用的计算方法，可选值：
                'eyes' - 使用眼角点
                'eyebrows' - 使用眉毛点
                'mouth' - 使用嘴角点
                'robust' - 使用多种方法的中位数（推荐）
    
    Returns:
        roll_angle: 估计的 roll 角度，单位为度
    """
    angles = []
    
    if method == 'eyes' or method == 'robust':
        # 使用眼角点 (左眼外角：33，右眼外角：263)
        left_eye = landmarks[33, :2]
        right_eye = landmarks[263, :2]
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        eye_angle = math.degrees(math.atan2(dy, dx))
        angles.append(eye_angle)
    
    if method == 'eyebrows' or method == 'robust':
        # 使用眉毛点 (左眉头：70，右眉头：300)
        left_eyebrow = landmarks[70, :2]
        right_eyebrow = landmarks[300, :2]
        dx = right_eyebrow[0] - left_eyebrow[0]
        dy = right_eyebrow[1] - left_eyebrow[1]
        eyebrow_angle = math.degrees(math.atan2(dy, dx))
        angles.append(eyebrow_angle)
    
    if method == 'mouth' or method == 'robust':
        # 使用嘴角点 (左嘴角：61，右嘴角：291)
        left_mouth = landmarks[61, :2]
        right_mouth = landmarks[291, :2]
        dx = right_mouth[0] - left_mouth[0]
        dy = right_mouth[1] - left_mouth[1]
        mouth_angle = math.degrees(math.atan2(dy, dx))
        angles.append(mouth_angle)
    
    # 添加额外的计算方法 - 使用对称点对
    if method == 'robust':
        # 额外点对1: 脸颊点 (左颊：207，右颊：427)
        left_cheek = landmarks[207, :2]
        right_cheek = landmarks[427, :2]
        dx = right_cheek[0] - left_cheek[0]
        dy = right_cheek[1] - left_cheek[1]
        cheek_angle = math.degrees(math.atan2(dy, dx))
        angles.append(cheek_angle)
        
        # 额外点对2: 下巴点 (下巴左：208，下巴右：428)
        left_jaw = landmarks[208, :2]
        right_jaw = landmarks[428, :2]
        dx = right_jaw[0] - left_jaw[0]
        dy = right_jaw[1] - left_jaw[1]
        jaw_angle = math.degrees(math.atan2(dy, dx))
        angles.append(jaw_angle)
    
    if method == 'robust':
        # 过滤异常值: 首先剔除异常点
        if len(angles) >= 3:
            angles = np.array(angles)
            median = np.median(angles)
            # 只保留在中位数附近的角度 (30度内)
            filtered_angles = angles[np.abs(angles - median) < 30]
            if len(filtered_angles) > 0:
                return np.median(filtered_angles)
            
        # 使用中位数
        return np.median(angles)
    else:
        # 使用单一方法
        return angles[0]

def estimate_face_pose_from_mediapipe(landmarks):
    """
    根据 MediaPipe 关键点估计人脸姿态（roll、yaw、pitch）
    
    Args:
        landmarks: MediaPipe 的完整人脸关键点
    
    Returns:
        roll, yaw, pitch: 姿态角度，单位为度
        confidence: 估计的可靠性
    """
    # 使用多种方法计算 roll 角度，取中位数
    roll = calculate_roll_angle(landmarks, method='robust')
    
    # 计算 yaw (偏航角) - 使用鼻子和两眼的相对位置
    # 眼角点 (左眼外角：33，右眼外角：263, 鼻尖: 4)
    left_eye = landmarks[33, :2]
    right_eye = landmarks[263, :2]
    nose_tip = landmarks[4, :2]
    
    # 计算眼睛中点
    eye_center = (left_eye + right_eye) / 2
    
    # 计算两眼间距
    eye_distance = np.linalg.norm(right_eye - left_eye)
    
    # 计算鼻尖与眼睛中点的水平距离
    nose_offset = nose_tip[0] - eye_center[0]
    
    # 改进的 yaw 估计 (使用脸部对称性)
    # 左右脸宽度比
    left_half = np.sum(landmarks[:, 0] < eye_center[0])
    right_half = np.sum(landmarks[:, 0] > eye_center[0])
    face_symmetry = min(left_half, right_half) / max(left_half, right_half)
    
    # 根据对称性调整 yaw 系数
    yaw_base = nose_offset / (eye_distance / 2) * 45
    # 当脸部不对称时，增强 yaw 估计
    if face_symmetry < 0.8:
        yaw_factor = 1.2
    else:
        yaw_factor = 1.0
    
    yaw = yaw_base * yaw_factor
    
    # 计算 pitch (俯仰角) - 使用鼻子和眼睛的垂直位置
    nose_eye_vertical = nose_tip[1] - eye_center[1]
    face_height = eye_distance * 1.5  # 估计脸部高度
    
    # 根据鼻尖的垂直位置估计 pitch 角度
    # 当鼻尖高于眼睛，pitch 为负值（抬头）；低于眼睛，pitch 为正值（低头）
    pitch = nose_eye_vertical / face_height * 45
    
    # 使用 3D 关键点增强 yaw 和 pitch 估计
    if landmarks.shape[1] == 3 and not np.isnan(landmarks[0, 2]):
        # 如果有可用的 z 坐标，用它增强 yaw 和 pitch 估计
        left_eye_z = landmarks[33, 2]
        right_eye_z = landmarks[263, 2]
        nose_z = landmarks[4, 2]
        
        # 眼睛中点的 z 坐标
        eye_center_z = (left_eye_z + right_eye_z) / 2
        
        # z 方向上鼻尖到眼睛中点的距离
        nose_eye_z = nose_z - eye_center_z
        
        # 调整 yaw 和 pitch 估计
        if abs(nose_eye_z) > 1e-6:  # 避免除以零
            # 调整 yaw 估计，考虑 z 坐标
            yaw_z = math.degrees(math.atan2(nose_offset, abs(nose_eye_z))) * 0.7
            yaw = (yaw + yaw_z) / 2  # 结合两种估计
            
            # 调整 pitch 估计，考虑 z 坐标
            pitch_z = math.degrees(math.atan2(nose_eye_vertical, abs(nose_eye_z))) * 0.7
            pitch = (pitch + pitch_z) / 2  # 结合两种估计
    
    # 确保角度在合理范围内
    roll = np.clip(roll, -180, 180)
    yaw = np.clip(yaw, -90, 90)
    pitch = np.clip(pitch, -90, 90)
    
    # 估计可靠性（置信度）
    # 根据眼睛和鼻子等关键点的可靠性进行估计
    confidence = 0.9  # MediaPipe 通常提供高质量关键点
    
    # 检测是否为闭眼状态
    left_eye_top = landmarks[159, 1]
    left_eye_bottom = landmarks[145, 1]
    right_eye_top = landmarks[386, 1]
    right_eye_bottom = landmarks[374, 1]
    
    left_eye_height = abs(left_eye_top - left_eye_bottom)
    right_eye_height = abs(right_eye_top - right_eye_bottom)
    
    # 如果眼睛高度很小，可能是闭眼状态，降低置信度
    if (left_eye_height < eye_distance * 0.03) or (right_eye_height < eye_distance * 0.03):
        confidence *= 0.8
        print("检测到闭眼状态，降低姿态估计置信度")
    
    return roll, yaw, pitch, confidence

def check_face_orientation(landmarks):
    """
    检查人脸朝向是否为正面
    
    Args:
        landmarks: MediaPipe 的完整人脸关键点
    
    Returns:
        is_frontal: 是否为正面人脸
        orientation: 人脸朝向描述 ('frontal', 'profile', 'tilted', 'inverted')
        confidence: 判断的置信度
    """
    # 计算姿态角度
    roll, yaw, pitch, confidence = estimate_face_pose_from_mediapipe(landmarks)
    
    # 定义判断标准 - 降低阈值以更敏感地检测异常方向
    is_frontal = True
    orientation = 'frontal'
    
    # 检查是否严重倾斜（roll角度）- 降低阈值
    if abs(roll) > 20:  # 从30度降到20度
        is_frontal = False
        orientation = 'tilted'
        confidence *= 0.9
    
    # 检查是否侧脸（yaw角度）- 降低阈值
    if abs(yaw) > 25:  # 从30度降到25度
        is_frontal = False
        orientation = 'profile'
        confidence *= 0.85
    
    # 检查是否抬头/低头（pitch角度）
    if abs(pitch) > 30:
        is_frontal = False
        if orientation == 'frontal':
            orientation = 'pitched'
        else:
            orientation += '_pitched'
        confidence *= 0.9
    
    # 检查是否倒置 - 降低阈值
    if abs(roll) > 140:  # 从150度降到140度
        is_frontal = False
        orientation = 'inverted'
        confidence *= 0.8
    
    # 组合处理，检测非常极端的角度
    if abs(roll) > 40 and abs(yaw) > 35:
        is_frontal = False
        orientation = 'extreme_pose'
        confidence *= 0.7
    
    return is_frontal, orientation, confidence

def analyze_face_symmetry(landmarks):
    """
    分析人脸对称性，用于辅助判断人脸朝向
    
    Args:
        landmarks: MediaPipe 的完整人脸关键点
    
    Returns:
        symmetry_score: 对称性得分 (0-1，越接近1越对称)
        is_symmetric: 是否对称
    """
    # 获取鼻梁点作为近似的脸部中线
    nose_bridge = landmarks[[168, 6, 197, 195, 5], :2]
    
    # 计算鼻梁中线的平均x坐标
    mid_x = np.mean(nose_bridge[:, 0])
    
    # 计算左右两侧点的数量和平均距离
    left_points = landmarks[landmarks[:, 0] < mid_x]
    right_points = landmarks[landmarks[:, 0] > mid_x]
    
    # 点数对称性
    count_ratio = min(len(left_points), len(right_points)) / max(len(left_points), len(right_points))
    
    # 计算到中心线的平均距离
    if len(left_points) > 0 and len(right_points) > 0:
        left_avg_dist = np.mean(abs(left_points[:, 0] - mid_x))
        right_avg_dist = np.mean(abs(right_points[:, 0] - mid_x))
        
        # 距离对称性
        dist_ratio = min(left_avg_dist, right_avg_dist) / max(left_avg_dist, right_avg_dist)
    else:
        dist_ratio = 0
    
    # 综合对称性得分 (70%点数比例 + 30%距离比例)
    symmetry_score = count_ratio * 0.7 + dist_ratio * 0.3
    
    # 判断是否对称
    is_symmetric = symmetry_score > 0.75
    
    return symmetry_score, is_symmetric

def rotate_image_to_correct_orientation(image_np, landmarks, max_cumulative_rotation=60, validate_rotation=True):
    """
    根据 MediaPipe 关键点旋转图像以校正人脸朝向，包含累计旋转跟踪和效果验证
    
    Args:
        image_np: 输入图像的 numpy 数组
        landmarks: MediaPipe 的完整人脸关键点
        max_cumulative_rotation: 最大累计旋转角度，默认60度
        validate_rotation: 是否验证旋转效果
    
    Returns:
        rotated_image: 旋转后的图像
        rotation_matrix: 旋转变换矩阵
        success: 是否成功旋转
    """
    h, w = image_np.shape[:2]
    
    # 估计姿态角度
    roll, yaw, pitch, confidence = estimate_face_pose_from_mediapipe(landmarks)
    print(f"开始估计人脸姿态...")
    print(f"估计的姿态角度: roll={roll:.1f}°, yaw={yaw:.1f}°, pitch={pitch:.1f}°, 置信度={confidence:.2f}")
    
    # 分析人脸对称性，辅助判断侧脸
    symmetry_score, is_symmetric = analyze_face_symmetry(landmarks)
    if not is_symmetric and abs(yaw) < 25:
        # 如果检测到不对称，但yaw不大，调整yaw估计
        yaw_adjustment = 15 if yaw >= 0 else -15
        yaw = yaw + yaw_adjustment
        print(f"基于对称性分析调整yaw角度: {yaw:.1f}°, 对称性分数: {symmetry_score:.2f}")
    
    # 计算人脸中心点
    face_center = np.mean(landmarks[:, :2], axis=0)
    center = (int(face_center[0]), int(face_center[1]))
    
    # 确定旋转角度 - 优化旋转策略
    rotation_angle = 0
    
    # 检测倒置的情况 - 改进检测方法
    is_inverted = False
    
    # 方法1: 检查roll角度是否接近180度
    if abs(roll) > 140:
        is_inverted = True
        print(f"基于roll角度({roll:.1f}°)检测到面部倒置")
    
    # 方法2: 检查特殊关键点的垂直位置
    # 眼睛应该在鼻子上方，嘴巴应该在鼻子下方
    eyes_y = (landmarks[33, 1] + landmarks[263, 1]) / 2  # 眼睛中心y坐标
    nose_y = landmarks[4, 1]  # 鼻尖y坐标
    mouth_y = (landmarks[61, 1] + landmarks[291, 1]) / 2  # 嘴巴中心y坐标
    
    # 正常脸: 眼睛y < 鼻子y < 嘴巴y
    # 倒置脸: 嘴巴y < 鼻子y < 眼睛y
    if eyes_y > nose_y > mouth_y:
        is_inverted = True
        print(f"基于关键点位置检测到面部倒置 (眼睛y: {eyes_y:.1f}, 鼻子y: {nose_y:.1f}, 嘴巴y: {mouth_y:.1f})")
    
    # 如果已经确定是倒置的，直接旋转180度
    if is_inverted:
        print("检测到面部完全倒置，直接旋转180度")
        rotation_angle = 180
        
        # 倒置情况下不受max_cumulative_rotation限制
        max_cumulative_rotation = max(180, max_cumulative_rotation)
    else:
        # 如果姿态估计结果不可靠，谨慎进行旋转
        if confidence < 0.5:
            print(f"姿态估计结果可靠性低(置信度={confidence:.2f})，谨慎旋转")
            # 降低阈值，但仍然进行必要的旋转
            roll_threshold = 20  # 正常10度
            yaw_threshold = 35   # 正常45度
        else:
            roll_threshold = 10
            yaw_threshold = 25   # 降低阈值，从45度降到25度
        
        # 优先处理roll角度
        if abs(roll) > roll_threshold:
            rotation_angle = -roll  # 负号是为了校正倾斜
            
            # 处理极端情况：倒置
            if abs(roll) > 140:  # 降低阈值，从165度降到140度
                # 如果接近180度倒置，直接旋转180度
                rotation_angle = 180 if roll > 0 else -180
                print(f"检测到面部倒置 (roll={roll:.1f}°)，旋转180度")
                # 倒置情况不受max_cumulative_rotation限制
                max_cumulative_rotation = max(180, max_cumulative_rotation)
            else:
                print(f"检测到面部倾斜 (roll={roll:.1f}°)，旋转{rotation_angle:.1f}度")
        
        # 处理侧脸（基于yaw角度）- 降低阈值
        if abs(yaw) > yaw_threshold:
            # 根据yaw的正负决定旋转方向
            # 平滑处理：角度越大，旋转越接近90度
            yaw_factor = min(abs(yaw) / 60, 1.0)  # 60度以上按满额计算
            add_angle = (90 if yaw > 0 else -90) * yaw_factor
            
            rotation_angle += add_angle
            print(f"检测到侧脸 (yaw={yaw:.1f}°)，增加旋转{add_angle:.1f}度")
    
    # 计算总旋转角度
    combined_angle = rotation_angle
    
    # 添加安全检查：如果同时有roll和yaw校正，检查是否会导致过度旋转
    if abs(roll) > 30 and abs(yaw) > 30:
        print(f"检测到同时存在较大的roll({roll:.1f}°)和yaw({yaw:.1f}°)，谨慎处理旋转角度...")
        # 如果roll和yaw角度同时较大且异号，可能会导致过度旋转
        if np.sign(roll) != np.sign(yaw) and abs(combined_angle) > 90:
            # 降低旋转幅度，避免生成倒置人脸
            old_angle = combined_angle
            # 限制在90度以内
            combined_angle = np.sign(combined_angle) * min(abs(combined_angle), 90)
            print(f"防止过度旋转: 从{old_angle:.1f}°调整为{combined_angle:.1f}°")
            rotation_angle = combined_angle
    
    # 如果总旋转角度很大，可能是倒置脸，调整为180度附近
    # 将阈值从150提高到165，更保守地应用180度旋转
    if abs(combined_angle) > 165 and abs(combined_angle) < 180:
        # 附加检查：确保确实是倒置脸而不是侧脸的误判
        if is_inverted or (abs(roll) > 120 and confidence > 0.7):
            rotation_angle = 180 if combined_angle > 0 else -180
            print(f"旋转角度({combined_angle:.1f}°)接近180度且确认需要旋转，调整为{rotation_angle}度")
            # 倒置情况不受max_cumulative_rotation限制
            max_cumulative_rotation = max(180, max_cumulative_rotation)
        else:
            # 如果不确定是倒置脸，降低旋转幅度，避免过度旋转
            print(f"旋转角度({combined_angle:.1f}°)接近180度但不确定是否需要完全翻转，降低旋转幅度")
            rotation_angle = np.sign(combined_angle) * 135  # 降低到135度
    
    # 限制最大旋转角度 (只在非倒置情况下)
    if abs(rotation_angle) < 170 and abs(rotation_angle) > max_cumulative_rotation:
        old_angle = rotation_angle
        rotation_angle = np.sign(rotation_angle) * max_cumulative_rotation
        print(f"旋转角度 {old_angle:.1f}° 超过限制，调整为 {rotation_angle:.1f}°")
    
    # 如果需要旋转的角度太小，直接返回原图
    if abs(rotation_angle) < 5:
        print("旋转角度太小 (<5°)，保持原始图像")
        # 创建恒等变换矩阵
        rotation_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        return image_np, rotation_matrix, False
    
    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    
    # 确保旋转后不裁剪图像的任何部分
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
    
    # 如果需要验证旋转效果 - 针对倒置情况修改验证逻辑
    if validate_rotation and not is_inverted:
        try:
            # 处理旋转后的图像以验证旋转效果
            from .face_corrector import multi_scale_detect
            
            # 在旋转后的图像上重新检测人脸
            rotated_faces, rotated_method, rotated_landmarks_list = multi_scale_detect(rotated_image, "mediapipe", 0.5)
            
            # 如果检测到人脸，验证旋转是否改善了姿态
            if rotated_faces and len(rotated_faces) > 0 and rotated_landmarks_list and len(rotated_landmarks_list) > 0:
                # 获取旋转后的最大人脸
                rotated_landmarks = rotated_landmarks_list[np.argmax([face.w * face.h for face in rotated_faces])]
                
                # 估计旋转后的姿态
                r_roll, r_yaw, r_pitch, r_confidence = estimate_face_pose_from_mediapipe(rotated_landmarks)
                
                print(f"旋转后姿态：roll={r_roll:.1f}°, yaw={r_yaw:.1f}°, pitch={r_pitch:.1f}°")
                
                # 判断旋转是否有效 - 关注roll和yaw的改善
                roll_improved = abs(r_roll) < abs(roll) * 0.7  # roll角度减少至少30%
                yaw_improved = abs(r_yaw) < abs(yaw) * 0.7     # yaw角度减少至少30%
                
                # 特殊情况：如果yaw或roll初始值很大，需要显著改善
                if abs(roll) > 30 or abs(yaw) > 30:
                    is_improved = roll_improved or yaw_improved
                else:
                    # 对于较小角度，更严格的判断
                    is_improved = (abs(r_roll) < 15 and abs(r_yaw) < 20)
                
                # 一般性判断：总体姿态是否更好
                overall_improved = (abs(r_roll) + abs(r_yaw)) < (abs(roll) + abs(yaw)) * 0.8
                
                # 对于大角度旋转(>90度)，使用不同的验证标准
                if abs(rotation_angle) > 90:
                    # 只要不是完全倒置就算是改善
                    is_improved = abs(r_roll) < 140 
                    overall_improved = True
                    print(f"大角度旋转({rotation_angle:.1f}°)，使用宽松验证标准")
                
                if not is_improved and not overall_improved:
                    print(f"旋转后姿态未改善，恢复原始图像")
                    return image_np, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), False
        except Exception as e:
            print(f"验证旋转效果时出错: {e}")
            # 错误处理：如果验证过程出错，仍然继续使用旋转后的图像
    elif is_inverted:
        print(f"面部倒置情况下跳过旋转效果验证，直接应用180度旋转")
    
    print(f"成功旋转图像，角度为{rotation_angle:.1f}度，新尺寸为{new_w}x{new_h}")
    return rotated_image, rotation_matrix, True

def mediapipe_to_insightface_landmarks(mp_landmarks):
    """
    将 MediaPipe 关键点转换为与 InsightFace 兼容的格式
    
    Args:
        mp_landmarks: MediaPipe 的完整人脸关键点
        
    Returns:
        insightface_kps: 与 InsightFace 兼容的 5 点关键点
    """
    # MediaPipe 关键点索引：
    # - 左眼：133 (左眼中心)
    # - 右眼：362 (右眼中心)
    # - 鼻尖：4
    # - 左嘴角：61
    # - 右嘴角：291
    
    # InsightFace 关键点顺序：左眼，右眼，鼻尖，左嘴角，右嘴角
    insightface_kps = np.array([
        mp_landmarks[133, :2],  # 左眼
        mp_landmarks[362, :2],  # 右眼
        mp_landmarks[4, :2],    # 鼻尖
        mp_landmarks[61, :2],   # 左嘴角
        mp_landmarks[291, :2]   # 右嘴角
    ])
    
    return insightface_kps

def get_rotated_face_landmarks(face, landmarks, rotation_matrix):
    """
    获取旋转后的人脸关键点
    
    Args:
        face: 原始的 Face 对象
        landmarks: MediaPipe 的完整人脸关键点
        rotation_matrix: 旋转变换矩阵
        
    Returns:
        rotated_face: 更新了关键点的 Face 对象
    """
    # 将关键点转换为齐次坐标
    landmarks_homogeneous = np.hstack([landmarks[:, :2], np.ones((landmarks.shape[0], 1))])
    
    # 应用旋转矩阵
    rotated_landmarks = np.dot(landmarks_homogeneous, rotation_matrix.T)
    
    # 获取旋转后的 5 点关键点
    rotated_kps = mediapipe_to_insightface_landmarks(rotated_landmarks)
    
    # 计算旋转后的边界框
    x_min = int(np.min(rotated_landmarks[:, 0]))
    y_min = int(np.min(rotated_landmarks[:, 1]))
    x_max = int(np.max(rotated_landmarks[:, 0]))
    y_max = int(np.max(rotated_landmarks[:, 1]))
    
    # 创建新的 Face 对象
    rotated_face = Face(
        bbox=[x_min, y_min, x_max, y_max],
        landmarks=rotated_kps,
        det_score=face.det_score
    )
    
    # 存储完整的旋转后关键点
    rotated_face.mediapipe_landmarks = rotated_landmarks
    
    return rotated_face 