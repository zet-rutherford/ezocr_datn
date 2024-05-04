from app.gvision.recognition.vietocr_recognition.vietocr.tool.predictor import Predictor
from app.gvision.recognition.vietocr_recognition.vietocr.tool.config import Cfg
from PIL import Image


class TextRecognizerVGGTrans():
    def __init__(self, device_id=0):
        config = Cfg.load_config_from_file('app/gvision/recognition/vietocr_recognition/vgg-seq2seq.yml')
        if device_id >= 0:
            config['device'] = 'cuda:{0}'.format(str(device_id))
        else:
            config['device'] = 'cpu'

        self.detector = Predictor(config)
        print('>> [VIETOCR: DEVICE_ID = {0}] loaded checkpoint vietocr vgg transformer text recognition'.format(device_id), flush=True)


    def recog(self, list_imgs_crop):
        list_imgs_crop_cvt = []
        for it in list_imgs_crop:
            list_imgs_crop_cvt.append(Image.fromarray(it))
        text, proba = self.detector.predict_batch(list_imgs_crop_cvt, return_prob=True)
        return text, proba
    
    def recog_single(self, crop_img):
        crop_img = Image.fromarray(crop_img)
        text, proba = self.detector.predict(crop_img, return_prob=True)
        return text, proba


