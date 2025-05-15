import torch
import numpy as np
from .utils import warp_face_back

class FacePaster:
    """人脸贴回节点，将处理后的人脸贴回原始图像"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "face_image": ("IMAGE",),
                "warp_matrix": ("WARP_MATRIX",),  # 变换矩阵
                "use_mask": (["是", "否"], {"default": "是"}),
            },
            "optional": {
                "face_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paste_face"
    CATEGORY = "image/face"
    
    def paste_face(self, original_image, face_image, warp_matrix, use_mask="是", face_mask=None):
        """将修复后的人脸贴回原图"""
        # 确保图像是正确的格式
        if isinstance(original_image, torch.Tensor):
            if len(original_image.shape) == 4:  # 批次维度
                image_np = original_image[0].cpu().numpy()
            else:
                image_np = original_image.cpu().numpy()
            
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = original_image
        
        if isinstance(face_image, torch.Tensor):
            if len(face_image.shape) == 4:  # 批次维度
                face_np = face_image[0].cpu().numpy()
            else:
                face_np = face_image.cpu().numpy()
            
            if face_np.max() <= 1.0:
                face_np = (face_np * 255).astype(np.uint8)
        else:
            face_np = face_image
        
        # 处理蒙版
        mask_np = None
        if use_mask == "是" and face_mask is not None:
            if isinstance(face_mask, torch.Tensor):
                if len(face_mask.shape) == 4:  # 批次维度
                    mask_np = face_mask[0].cpu().numpy()
                else:
                    mask_np = face_mask.cpu().numpy()
            else:
                mask_np = face_mask
        
        # 打印调试信息
        print(f"图像尺寸: {image_np.shape}")
        print(f"人脸尺寸: {face_np.shape}")
        if mask_np is not None:
            print(f"蒙版尺寸: {mask_np.shape}")
        print(f"变换矩阵: {warp_matrix}")
        print(f"使用蒙版: {use_mask}")
        
        # 贴回人脸
        try:
            result = warp_face_back(image_np, face_np, warp_matrix, mask_np)
            
            # 转回tensor
            result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0)
            if len(result_tensor.shape) == 3:  # 添加批次维度
                result_tensor = result_tensor.unsqueeze(0)
            
            return (result_tensor,)
        except Exception as e:
            print(f"贴回人脸失败: {e}")
            # 返回原图
            result_tensor = torch.from_numpy(image_np.astype(np.float32) / 255.0)
            if len(result_tensor.shape) == 3:  # 添加批次维度
                result_tensor = result_tensor.unsqueeze(0)
            
            return (result_tensor,)