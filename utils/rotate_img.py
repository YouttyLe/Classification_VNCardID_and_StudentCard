import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image
from model.model_yolov8.detection import crop_cccd_tsv
# Đường dẫn đến tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'tesserat\tesseract.exe'

def resize_image(image, scale_factor=1.5):
    """
    Hàm phóng to ảnh để cải thiện độ chính xác của OCR.

    Args:
        image (numpy.ndarray): Ảnh dưới dạng mảng numpy.
        scale_factor (float): Tỉ lệ phóng to ảnh (mặc định là 1.5).

    Returns:
        numpy.ndarray: Ảnh đã được phóng to.
    """
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized_image


def rotate_image(image):
    """
    Hàm xoay ảnh dựa trên góc được phát hiện bởi Tesseract,
    không tạo vùng đen và giữ nguyên kích thước.

    Args:
        image (numpy.ndarray): Ảnh dưới dạng mảng numpy.

    Returns:
        numpy.ndarray: Ảnh đã xoay hoặc ảnh gốc nếu không tìm thấy góc.
    """
    if image is None:
        return None

    # Chuyển ảnh sang grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Phóng to ảnh để cải thiện độ chính xác của Tesseract
    resized_gray = resize_image(img_gray, 1.5)

    # Lấy góc xoay từ Tesseract
    angle = get_orientation(resized_gray)

    # Nếu có góc xoay, thực hiện xoay ảnh
    if angle != 0:
        image = _rotate_image(image, -angle)

    return image


def get_orientation(img_gray):
    """
    Hàm lấy góc xoay của ảnh từ Tesseract.

    Args:
        img_gray (numpy.ndarray): Ảnh grayscale.

    Returns:
        int: Góc xoay của ảnh.
    """
    try:
        data = pytesseract.image_to_osd(img_gray, output_type=Output.DICT)
        angle = data['rotate']
        return angle
    except:
        return 0  # Trả về 0 nếu không thể lấy được góc xoay


def _rotate_image(image, angle):
    """
    Hàm xoay ảnh mà không tạo vùng đen và giữ nguyên kích thước.

    Args:
        image (numpy.ndarray): Ảnh cần xoay.
        angle (float): Góc xoay (độ).

    Returns:
        numpy.ndarray: Ảnh đã xoay.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Tính toán ma trận xoay
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Tính toán cos và sin của góc
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])

    # Tính toán kích thước ảnh mới (bao gồm cả phần xoay)
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    # Điều chỉnh ma trận xoay để dịch chuyển tâm xoay
    M[0, 2] += bound_w / 2 - center[0]
    M[1, 2] += bound_h / 2 - center[1]

    # Xoay ảnh
    rotated = cv2.warpAffine(image, M, (bound_w, bound_h))
    return rotated