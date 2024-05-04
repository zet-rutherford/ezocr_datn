import logging
import os
import pathlib
import sys
from datetime import datetime
import json
import requests

from dotenv import load_dotenv

# env = load_dotenv()
env = load_dotenv('env.example')

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent

LOG_DIR = PACKAGE_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / 'app.log'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

FORMATTER = logging.Formatter("%(message)s")


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_logger(*, logger_name):
    """Get logger with prepared handlers."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(FORMATTER)

    logger.addHandler(file_handler)
    logger.addHandler(get_console_handler())
    logger.propagate = False

    return logger

logger = get_logger(logger_name=__name__)


class ApplicationConfig:
    
    # Landmark paper
    LANDMARK_YOLOV8_PATH = os.environ.get('LANDMARK_YOLOV8_PATH')
    LANDMARK_YOLOV8_IMAGE_SIZE = int(os.environ.get('LANDMARK_YOLOV8_IMAGE_SIZE'))
    LANDMARK_YOLOV8_IOU_THRES = float(os.environ.get('LANDMARK_YOLOV8_IOU_THRES'))
    LANDMARK_YOLOV8_CONF_THRES = float(os.environ.get('LANDMARK_YOLOV8_CONF_THRES'))

    # PaddlOCR for detect line text
    PADDLEOCR_DETECTION_MODEL_DIR = os.environ.get('PADDLEOCR_DETECTION_MODEL_DIR')
    PADDLEOCR_CLS_MODEL_DIR = os.environ.get('PADDLEOCR_CLS_MODEL_DIR')
    PADDLEOCR_REC_MODEL_DIR = os.environ.get('PADDLEOCR_REC_MODEL_DIR')    # model ocr handwritting

    # device < 0: cpu, > 0: gpu
    DEVICE_GPU = int(os.environ.get('DEVICE_GPU'))

config = ApplicationConfig()
