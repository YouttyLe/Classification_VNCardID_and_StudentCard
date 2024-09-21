import cv2

def crop_info(img):
    crop_info= img[169:440, 180:640]
    gray = cv2.cvtColor(crop_info, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    gray = cv2.resize(thresholded,(600, 600))
    # gray= cv2.add(gray, 30)
    id_crop = gray[0:117, 80:600]
    name_crop = gray[150:233, 0:600]
    date_crop = gray[215:290, 225:600]
    sex_crop = gray[270:365, 150:235]
    national_crop = gray[270:370, 430:600]
    address_crop = gray[400:465, 0:600]
    home_crop_1 = gray[455:530, 325:600]
    home_crop_2 = gray[509:600, 0:600]
    return [id_crop, name_crop, date_crop, sex_crop, national_crop, address_crop, home_crop_1, home_crop_2]

