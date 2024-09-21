import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'D:\tesserat\tesseract.exe'
def extract_text_from_image(image_path):
    """
    Hàm trích xuất thông tin từ ảnh CCCD đã cắt.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 440))

    id_crop = img[168:222, 250:600]
    name_crop = img[235:274, 190:600]
    sex_crop = img[290:330, 293:360]
    national_crop = img[290:330, 515:640]
    address_crop = img[349:375, 170:640]
    home_crop_1 = img[374:405, 435:640]
    home_crop_2 = img[400:440, 180:640]

    extracted_info = {}
    extracted_info['id'] = pytesseract.image_to_string(id_crop, lang='vie').strip()
    extracted_info['name'] = pytesseract.image_to_string(name_crop, lang='vie').strip()
    extracted_info['sex'] = pytesseract.image_to_string(sex_crop, lang='vie').strip()
    extracted_info['nationality'] = pytesseract.image_to_string(national_crop, lang='vie').strip()
    extracted_info['address'] = pytesseract.image_to_string(address_crop, lang='vie').strip()
    extracted_info['hometown'] = pytesseract.image_to_string(home_crop_1, lang='vie').strip() + " " + pytesseract.image_to_string(home_crop_2, lang='vie').strip()

    return extracted_info