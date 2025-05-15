import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

class ImageProcessor:
    """图像处理工具类"""
    
    def process_face(self, image, mask, face_info, 
                    correction_strength=0.5, 
                    enable_rotation=True,
                    enable_symmetry=False,
                    eye_enlargement=0.0) -> Tuple[np.ndarray, np.ndarray]:
        """处理人脸图像"""
        # 创建输出图像的副本
        result_image = image.copy()
        result_mask = np.zeros_like(mask)
        
        # 如果人脸检测失败，返回原图
        if face_info["status"] != "success" or face_info["landmarks"] is None:
            return image, mask
        
        # 获取人脸区域
        x, y, w, h = face_info["face_rect"]
        
        # 扩大人脸区域
        padding = int(max(w, h) * 0.2)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        # 提取人脸区域
        face_region = image[y1:y2, x1:x2].copy()
        face_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
        cv2.fillConvexPoly(face_mask, self._get_face_contour(face_info, x1, y1), 255)
        
        # 应用校正
        if enable_rotation and abs(face_info["face_angle"]) > 1.0:
            # 旋转校正
            face_region, face_mask = self._rotate_correction(
                face_region, face_mask, face_info, x1, y1, correction_strength
            )
        
        if enable_symmetry:
            # 对称校正
            face_region = self._symmetry_correction(
                face_region, face_mask, face_info, x1, y1, correction_strength
            )
        
        if eye_enlargement > 0:
            # 眼睛放大
            face_region = self._enlarge_eyes(
                face_region, face_mask, face_info, x1, y1, eye_enlargement
            )
        
        # 将处理后的人脸区域放回原图
        # 创建alpha通道
        alpha = face_mask.astype(np.float32) / 255.0
        alpha = alpha[..., np.newaxis]
        
        # 将处理后的人脸区域与原图混合
        result_image[y1:y2, x1:x2] = (
            (1 - alpha) * result_image[y1:y2, x1:x2] + 
            alpha * face_region
        ).astype(np.uint8)
        
        # 更新mask
        result_mask[y1:y2, x1:x2] = face_mask
        
        return result_image, result_mask
    
    def _get_face_contour(self, face_info, offset_x=0, offset_y=0) -> np.ndarray:
        """获取人脸轮廓点"""
        if face_info["detection_method"] == "insightface":
            # InsightFace的轮廓点
            contour_indices = list(range(0, 33))  # 前33个点是轮廓
            landmarks = face_info["landmarks"]
            contour = landmarks[