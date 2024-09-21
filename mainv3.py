import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from model.model_yolov8.detection import crop_cccd_tsv
from utils.rotate_img import rotate_image
from utils.crop_textv2 import crop_info

pytesseract.pytesseract.tesseract_cmd = r'D:\tesserat\tesseract.exe'


img_path =('pic4.jpg')
detect_img = crop_cccd_tsv(img_path)

# Kiểm tra xem detect_img có phải là None không trước khi thực hiện xoay và cắt
if detect_img is not None:
    rotate_imgage = rotate_image(detect_img)
    crop = crop_info(rotate_imgage)

    for i in range(len(crop)):
        text = pytesseract.image_to_string(crop[i], lang='vie')
        print(text)
else:
    print("Không tìm thấy thẻ CCCD/TSV trong ảnh.")