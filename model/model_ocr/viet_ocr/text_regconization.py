import matplotlib.pyplot as plt
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


# config = Cfg.load_config_from_name('vgg_transformer')
# config['weights'] = 'model/model_ocr/viet_ocr/transformerocr.pth'  # Thay đổi đường dẫn nếu cần
# config['cnn']['pretrained']=False
# config['device'] = 'cpu'
# detector = Predictor(config)

def extract_text_from_image(image_path):
    """
    Hàm trích xuất văn bản từ ảnh bằng vietocr.
    """
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = 'model/model_ocr/viet_ocr/transformerocr.pth'  # Thay đổi đường dẫn nếu cần
    config['cnn']['pretrained'] = False
    config['device'] = 'cpu'
    detector = Predictor(config)

    img = Image.open(image_path)
    extracted_text = detector.predict(img)
    return extracted_text