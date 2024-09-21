# import os
# import cv2
# from ultralytics import YOLO
#
#
# def crop_cccd_tsv(image_path, output_folder_cccd='output/CCCD', output_folder_tsv='output/TSV', conf_threshold=0.90):
#     """
#     Hàm crop thẻ CCCD và TSV từ ảnh đầu vào và lưu vào thư mục tương ứng.
#
#     Args:
#         image_path (str): Đường dẫn đến ảnh đầu vào.
#         output_folder_cccd (str, optional): Tên thư mục lưu ảnh CCCD đã crop. Defaults to 'CCCD'.
#         output_folder_tsv (str, optional): Tên thư mục lưu ảnh TSV đã crop. Defaults to 'TSV'.
#         model_path (str, optional): Đường dẫn đến file mô hình YOLO. Defaults to 'best_YoloV8m_v1.pt'.
#         conf_threshold (float, optional): Ngưỡng confidence score để lọc kết quả. Defaults to 0.90.
#     """
#
#     global output_path
#     model = YOLO('Detection_cccd/best_finetuning_v2.pt')
#
#     Tạo thư mục output nếu chưa tồn tại
    # os.makedirs(output_folder_cccd, exist_ok=True)
    # os.makedirs(output_folder_tsv, exist_ok=True)
    #
    # original_filename = os.path.splitext(os.path.basename(image_path))[0]
    #
    # results = model.predict(image_path)
    #
    # for i, box in enumerate(results[0].boxes):
    #     class_name = model.names[int(box.cls)]
    #     conf = float(box.conf)  # Lấy confidence score
    #     if conf >= conf_threshold:
    #         if class_name in ['CCCD', 'TSV']:
    #             x1, y1, x2, y2 = map(int, box.xyxy[0])
    #             cropped_img = results[0].orig_img[y1:y2, x1:x2]
    #
    #             if class_name == 'TSV':
    #                 output_filename = f'{original_filename}_cropped_tsv_{i}.jpg'
    #                 output_path = os.path.join(output_folder_tsv, output_filename)
    #             elif class_name == 'CCCD':
    #                 output_filename = f'{original_filename}_cropped_cccd_{i}.jpg'
    #                 output_path = os.path.join(output_folder_cccd, output_filename)
    #
    #             cv2.imwrite(output_path, cropped_img)
    #
    # return output_path


# import os
# import cv2
# from ultralytics import YOLO
#
# def crop_cccd_tsv(image_path, output_folder_cccd='output/CCCD', output_folder_tsv='output/TSV', conf_threshold=0.90):
#     """
#     Hàm crop thẻ CCCD và TSV từ ảnh đầu vào và lưu vào thư mục tương ứng.
#     """
#     model = YOLO('model/model_yolov8/best_finetuning_v2.pt')
#
#     os.makedirs(output_folder_cccd, exist_ok=True)
#     os.makedirs(output_folder_tsv, exist_ok=True)
#
#     original_filename = os.path.splitext(os.path.basename(image_path))[0]
#     results = model.predict(image_path)
#
#     cropped_paths = []
#
#     for i, box in enumerate(results[0].boxes):
#         class_name = model.names[int(box.cls)]
#         conf = float(box.conf)
#         if conf >= conf_threshold:
#             if class_name in ['CCCD', 'TSV']:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cropped_img = results[0].orig_img[y1:y2, x1:x2]
#
#                 if class_name == 'TSV':
#                     output_filename = f'{original_filename}_cropped_tsv_{i}.jpg'
#                     output_path = os.path.join(output_folder_tsv, output_filename)
#                 elif class_name == 'CCCD':
#                     output_filename = f'{original_filename}_cropped_cccd_{i}.jpg'
#                     output_path = os.path.join(output_folder_cccd, output_filename)
#
#                 cv2.imwrite(output_path, cropped_img)
#                 cropped_paths.append(output_path)
#
#     return cropped_paths
#
from ultralytics import YOLO
import cv2


def crop_cccd_tsv(img_path, conf_threshold=0.90):
    """
    Hàm crop thẻ CCCD và TSV từ ảnh đầu vào và trả về ảnh đã crop.
    Nếu không detect được hoặc không crop được, trả về None.

    Args:
        img_path (str): Đường dẫn đến ảnh cần crop.
        conf_threshold (float, optional): Ngưỡng confidence score để lọc kết quả.
                                        Mặc định là 0.90.
    Returns:
        numpy.ndarray: Ảnh đã crop hoặc None nếu không tìm thấy thẻ.
    """
    model = YOLO('model/model_yolov8/best_yolov8m-seg.pt')
    try:
        results = model.predict(img_path)

        for i, box in enumerate(results[0].boxes):
            class_name = model.names[int(box.cls)]
            conf = float(box.conf)
            if conf >= conf_threshold:
                if class_name in ['CCCD', 'TSV']:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cropped_img = results[0].orig_img[y1:y2, x1:x2]
                    return cropped_img

        return None  # Trả về None nếu không tìm thấy thẻ phù hợp

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        return None  # Trả về None nếu có lỗi xảy ra




