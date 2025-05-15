import cv2
import numpy as np
import mediapipe as mp
import os
from typing import Dict, Any, List, Optional, Tuple

# 尝试导入dlib
try:
    import dlib
    has_dlib = True
except ImportError:
    has_dlib = False
    print("警告: 未找到dlib库，将使用备用人脸检测方法")

# 尝试导入insightface
try:
    import insightface
    from insightface.app import FaceAnalysis
    has_insightface = True
except ImportError:
    has_insightface = False
    print("警告: 未找到insightface库，将使用备用人脸检测方法")

class FaceDetector:
    """人脸检测工具类"""
    
    # 类变量用于缓存模型
    _face_analyzer = None
    _last_face_info = None
    _last_image_hash = None
    
    def __init__(self):
        # 初始化MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # 初始化dlib检测器
        if has_dlib:
            # 获取模型文件路径
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(current_dir, "models")
            
            # 如果models目录不存在，创建它
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
                print(f"创建models目录: {models_dir}")
            
            shape_predictor_path = os.path.join(models_dir, "shape_predictor_68_face_landmarks.dat")
            
            # 检查模型文件是否存在
            if not os.path.exists(shape_predictor_path):
                print(f"dlib模型文件不存在: {shape_predictor_path}")
                print("请从http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2下载并解压到models目录")
                self.dlib_detector = None
                self.dlib_predictor = None
            else:
                print(f"加载dlib模型: {shape_predictor_path}")
                self.dlib_detector = dlib.get_frontal_face_detector()
                self.dlib_predictor = dlib.shape_predictor(shape_predictor_path)
                print("dlib模型加载成功")
        else:
            self.dlib_detector = None
            self.dlib_predictor = None
    
    @classmethod
    def _get_face_analyzer(cls):
        """单例模式获取InsightFace分析器"""
        if cls._face_analyzer is None and has_insightface:
            try:
                model_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "models",
                    "insightface"
                )
                
                analyzer = FaceAnalysis(
                    root=model_dir,
                    providers=['CPUExecutionProvider'],
                    allowed_modules=['detection', 'recognition', 'genderage'],
                    name="buffalo_l"
                )
                analyzer.prepare(ctx_id=0, det_size=(640, 640))
                cls._face_analyzer = analyzer
                print("InsightFace模型加载成功")
            except Exception as e:
                print(f"InsightFace初始化失败: {str(e)}")
                return None
        return cls._face_analyzer
    
    def detect_face(self, image, mask=None, detection_method="auto") -> Dict[str, Any]:
        """检测图像中的人脸"""
        # 计算图像哈希值用于缓存
        image_hash = hash(image.tobytes())
        
        # 检查缓存
        if FaceDetector._last_image_hash == image_hash and FaceDetector._last_face_info is not None:
            print("使用缓存的人脸分析结果")
            return FaceDetector._last_face_info.copy()
        
        # 初始化face_info
        face_info = {
            "status": "error",
            "message": "未检测到人脸",
            "landmarks": None,
            "face_rect": None,
            "eyes_center": None,
            "left_eye": None,
            "right_eye": None,
            "nose": None,
            "mouth": None,
            "face_angle": 0.0,
            "confidence": 0.0,
            "gender": "unknown",
            "detection_method": "none"
        }
        
        # 按优先级尝试不同的检测方法
        detection_methods = []
        if detection_method == "insightface" and has_insightface:
            detection_methods = ["insightface"]
        elif detection_method == "mediapipe":
            detection_methods = ["mediapipe"]
        elif detection_method == "dlib" and has_dlib:
            detection_methods = ["dlib"]
        elif detection_method == "opencv":
            detection_methods = ["opencv"]
        elif detection_method == "auto":
            # 自动模式下按优先级尝试所有可用方法
            detection_methods = []
            if has_insightface:
                detection_methods.append("insightface")
            detection_methods.extend(["mediapipe", "dlib", "opencv"])
        
        # 按顺序尝试各种方法
        for method in detection_methods:
            if method == "insightface" and has_insightface:
                if self._detect_face_insightface(image, face_info):
                    break
            elif method == "mediapipe":
                if self._detect_face_mediapipe(image, face_info):
                    break
            elif method == "dlib" and has_dlib:
                if self._detect_face_dlib(image, face_info):
                    break
            elif method == "opencv":
                if self._detect_face_opencv(image, face_info):
                    break
        
        # 更新缓存
        FaceDetector._last_image_hash = image_hash
        FaceDetector._last_face_info = face_info.copy()
        
        return face_info
    
    def _detect_face_insightface(self, image, face_info):
        """使用InsightFace检测人脸"""
        try:
            analyzer = self._get_face_analyzer()
            if analyzer is None:
                return False
            
            # 确保图像是BGR格式
            if image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            faces = analyzer.get(image_rgb)
            
            if not faces:
                return False
            
            # 使用最大的人脸
            max_face = max(faces, key=lambda face: 
                (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
            
            bbox = max_face.bbox.astype(int)
            x, y, x2, y2 = bbox
            w, h = x2 - x, y2 - y
            
            # 获取关键点
            landmarks = max_face.landmark_2d_106 if hasattr(max_face, 'landmark_2d_106') else None
            
            if landmarks is not None and len(landmarks) > 0:
                # 计算眼睛位置
                left_eye = np.mean(landmarks[60:68], axis=0)  # 左眼关键点
                right_eye = np.mean(landmarks[68:76], axis=0)  # 右眼关键点
                eyes_center = (left_eye + right_eye) / 2
                
                # 计算鼻子和嘴巴位置
                nose = landmarks[86]  # 鼻尖
                mouth = np.mean(landmarks[96:106], axis=0)  # 嘴巴中心
                
                # 计算人脸角度
                eye_angle = np.degrees(np.arctan2(
                    right_eye[1] - left_eye[1],
                    right_eye[0] - left_eye[0]
                ))
                
                face_info.update({
                    "status": "success",
                    "message": "人脸检测成功",
                    "landmarks": landmarks,
                    "face_rect": (x, y, w, h),
                    "eyes_center": tuple(eyes_center.astype(int)),
                    "left_eye": tuple(left_eye.astype(int)),
                    "right_eye": tuple(right_eye.astype(int)),
                    "nose": tuple(nose.astype(int)),
                    "mouth": tuple(mouth.astype(int)),
                    "face_angle": eye_angle,
                    "confidence": float(max_face.det_score),
                    "gender": "male" if max_face.gender == 1 else "female",
                    "detection_method": "insightface"
                })
                
                print(f"InsightFace检测成功，置信度: {face_info['confidence']:.3f}")
                return True
            
        except Exception as e:
            print(f"InsightFace分析失败: {str(e)}")
        
        return False
    
    def _detect_face_mediapipe(self, image, face_info):
        """使用MediaPipe检测人脸"""
        try:
            # 确保图像是RGB格式
            if image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            results = self.face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                return False
            
            # 获取第一个人脸的关键点
            face_landmarks = results.multi_face_landmarks[0]
            
            # 转换关键点坐标
            h, w = image.shape[:2]
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append([x, y])
            
            landmarks = np.array(landmarks)
            
            # 计算人脸边界框
            x_min = np.min(landmarks[:, 0])
            y_min = np.min(landmarks[:, 1])
            x_max = np.max(landmarks[:, 0])
            y_max = np.max(landmarks[:, 1])
            
            # 计算眼睛位置
            left_eye = np.mean(landmarks[[33, 133, 157, 158, 159, 160, 161, 173, 246]], axis=0)
            right_eye = np.mean(landmarks[[263, 362, 384, 385, 386, 387, 388, 398, 466]], axis=0)
            eyes_center = (left_eye + right_eye) / 2
            
            # 计算鼻子和嘴巴位置
            nose = landmarks[1]  # 鼻尖
            mouth = np.mean(landmarks[[61, 185, 40, 39, 37, 0, 267, 269, 270, 409]], axis=0)  # 嘴巴中心
            
            # 计算人脸角度
            eye_angle = np.degrees(np.arctan2(
                right_eye[1] - left_eye[1],
                right_eye[0] - left_eye[0]
            ))
            
            face_info.update({
                "status": "success",
                "message": "人脸检测成功",
                "landmarks": landmarks,
                "face_rect": (x_min, y_min, x_max - x_min, y_max - y_min),
                "eyes_center": tuple(eyes_center.astype(int)),
                "left_eye": tuple(left_eye.astype(int)),
                "right_eye": tuple(right_eye.astype(int)),
                "nose": tuple(nose.astype(int)),
                "mouth": tuple(mouth.astype(int)),
                "face_angle": eye_angle,
                "confidence": 0.9,  # MediaPipe没有置信度，使用默认值
                "gender": "unknown",  # MediaPipe无法判断性别
                "detection_method": "mediapipe"
            })
            
            print("MediaPipe检测成功")
            return True
            
        except Exception as e:
            print(f"MediaPipe分析失败: {str(e)}")
        
        return False
    
    def _detect_face_dlib(self, image, face_info):
        """使用dlib检测人脸"""
        if not has_dlib or self.dlib_detector is None:
            return False
        
        try:
            # 转换为灰度图像
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 检测人脸
            faces = self.dlib_detector(gray)
            
            if not faces:
                return False
            
            # 使用最大的人脸
            max_face = max(faces, key=lambda face: face.area())
            
            # 获取人脸关键点
            shape = self.dlib_predictor(gray, max_face)
            landmarks = []
            for i in range(68):
                x = shape.part(i).x
                y = shape.part(i).y
                landmarks.append([x, y])
            
            landmarks = np.array(landmarks)
            
            # 计算人脸边界框
            x, y, w, h = max_face.left(), max_face.top(), max_face.width(), max_face.height()
            
            # 计算眼睛位置
            left_eye = np.mean(landmarks[36:42], axis=0)  # 左眼关键点
            right_eye = np.mean(landmarks[42:48], axis=0)  # 右眼关键点
            eyes_center = (left_eye + right_eye) / 2
            
            # 计算鼻子和嘴巴位置
            nose = landmarks[30]  # 鼻尖
            mouth = np.mean(landmarks[48:68], axis=0)  # 嘴巴中心
            
            # 计算人脸角度
            eye_angle = np.degrees(np.arctan2(
                right_eye[1] - left_eye[1],
                right_eye[0] - left_eye[0]
            ))
            
            face_info.update({
                "status": "success",
                "message": "人脸检测成功",
                "landmarks": landmarks,
                "face_rect": (x, y, w, h),
                "eyes_center": tuple(eyes_center.astype(int)),
                "left_eye": tuple(left_eye.astype(int)),
                "right_eye": tuple(right_eye.astype(int)),
                "nose": tuple(nose.astype(int)),
                "mouth": tuple(mouth.astype(int)),
                "face_angle": eye_angle,
                "confidence": 0.8,  # dlib没有置信度，使用默认值
                "gender": "unknown",  # dlib无法判断性别
                "detection_method": "dlib"
            })
            
            print("dlib检测成功")
            return True
            
        except Exception as e:
            print(f"dlib分析失败: {str(e)}")
        
        return False
    
    def _detect_face_opencv(self, image, face_info):
        """使用OpenCV检测人脸"""
        try:
            # 加载OpenCV人脸检测器
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
            eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            
            # 转换为灰度图像
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 检测人脸
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return False
            
            # 使用最大的人脸
            max_face = max(faces, key=lambda face: face[2] * face[3])
            x, y, w, h = max_face
            
            # 提取人脸区域
            roi_gray = gray[y:y+h, x:x+w]
            
            # 检测眼睛
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            if len(eyes) >= 2:
                # 按x坐标排序
                eyes = sorted(eyes, key=lambda eye: eye[0])
                
                # 计算眼睛中心点
                left_eye_x = x + eyes[0][0] + eyes[0][2] // 2
                left_eye_y = y + eyes[0][1] + eyes[0][3] // 2
                right_eye_x = x + eyes[1][0] + eyes[1][2] // 2
                right_eye_y = y + eyes[1][1] + eyes[1][3] // 2
                
                left_eye = (left_eye_x, left_eye_y)
                right_eye = (right_eye_x, right_eye_y)
                eyes_center = ((left_eye_x + right_eye_x) // 2, (left_eye_y + right_eye_y) // 2)
                
                # 计算人脸角度
                eye_angle = np.degrees(np.arctan2(
                    right_eye_y - left_eye_y,
                    right_eye_x - left_eye_x
                ))
                
                # 估计鼻子和嘴巴位置
                nose = (x + w // 2, y + h // 2)
                mouth = (x + w // 2, y + int(h * 0.7))
                
                face_info.update({
                    "status": "success",
                    "message": "人脸检测成功",
                    "landmarks": None,  # OpenCV没有详细的关键点
                    "face_rect": (x, y, w, h),
                    "eyes_center": eyes_center,
                    "left_eye": left_eye,
                    "right_eye": right_eye,
                    "nose": nose,
                    "mouth": mouth,
                    "face_angle": eye_angle,
                    "confidence": 0.7,  # OpenCV没有置信度，使用默认值
                    "gender": "unknown",  # OpenCV无法判断性别
                    "detection_method": "opencv"
                })
                
                print("OpenCV检测成功")
                return True
            
        except Exception as e:
            print(f"OpenCV分析失败: {str(e)}")
        
        return False