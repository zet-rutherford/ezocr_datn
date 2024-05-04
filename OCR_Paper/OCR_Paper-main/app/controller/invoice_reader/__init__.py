import torch
import sys
from app.extensions import config

# def get_sticker_landmark():
#     from app.gvision.landmark.Yolov8 import Yolov8_Landmark
#     return Yolov8_Landmark(model_dir=config.LANDMARK_YOLOV8_PATH,
#                            image_size=config.LANDMARK_YOLOV8_IMAGE_SIZE,
#                            conf=config.LANDMARK_YOLOV8_CONF_THRES,
#                            iou=config.LANDMARK_YOLOV8_IOU_THRES,
#                            device=config.DEVICE_GPU)

def get_paddleocr_detect_line():
    from app.gvision.recognition.paddle_recog.recognition import TextRecognitionPaddle
    return TextRecognitionPaddle(device=config.DEVICE_GPU)

def get_vietocr_reader():
    from app.gvision.recognition.vietocr_recognition.recognition import TextRecognizerVGGTrans
    return TextRecognizerVGGTrans(device_id=config.DEVICE_GPU)


def get_algorithm():
    from app.controller.invoice_reader.algorithm import Algorithm
    with torch.no_grad():
        landmark_detector = None
        paddleocr_text_detector = get_paddleocr_detect_line()
        vietocr_recognizer = get_vietocr_reader()

    return Algorithm(landmark_detector, paddleocr_text_detector, vietocr_recognizer)
