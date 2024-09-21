import os
import cv2
from utils.crop_detail_img import extract_text_from_image
from model.model_yolov8.detection import crop_cccd_tsv
from utils.rotate_img import rotate_image
from model.model_ocr.viet_ocr.text_regconization import extract_text_from_image

def find_non_black_area(image):
    """
    Hàm tìm vùng không phải màu đen trong ảnh.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 79, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
      largest_contour = max(contours, key=cv2.contourArea)
      x, y, w, h = cv2.boundingRect(largest_contour)
      return x, y, w, h
    else:
        return 0, 0, image.shape[1], image.shape[0]

def process_image(image_path, output_file):
    """
    Hàm xử lý một ảnh duy nhất: crop, xoay, trích xuất thông tin.
    """
    cropped_paths = crop_cccd_tsv(image_path)

    for cropped_path in cropped_paths:
        if cropped_path:
            # Xoay ảnh trước khi xử lý khác
            rotated_image = rotate_image(cropped_path)
            cv2.imwrite(cropped_path, rotated_image)

            # Xử lý cắt bỏ vùng đen nếu cần
            image = cv2.imread(cropped_path)
            x, y, w, h = find_non_black_area(image)
            cropped_image = image[y:y + h, x:x + w]
            cropped_image_path = 'cropped_cccd_image.jpg'
            cv2.imwrite(cropped_image_path, cropped_image)

            # Trích xuất văn bản từ ảnh đã cắt
            extracted_text = extract_text_from_image(cropped_image_path)

            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"Kết quả từ ảnh: {os.path.basename(image_path)}\n")
                f.write(f"{extracted_text}\n")
                f.write("-" * 20 + "\n")

if __name__ == "__main__":
    image_folder = 'D:\Detection_cccd\data_test'
    output_text_file = 'info.txt'

    if os.path.exists(output_text_file):
        os.remove(output_text_file)

    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, filename)
            process_image(image_path, output_text_file)