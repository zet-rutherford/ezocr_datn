from app.gvision.recognition.paddle_recog.pp_recognition.paddleocr import PaddleOCR
import cv2
import numpy as np
from app.extensions import config


class TextRecognitionPaddle:
    def __init__(self, device:int):
        if device < 0:
            self.ocr = PaddleOCR(det_db_unclip_ratio=2.2, use_gpu=False, use_angle_cls=True, det_model_dir=config.PADDLEOCR_DETECTION_MODEL_DIR, cls_model_dir=config.PADDLEOCR_CLS_MODEL_DIR, rec_model_dir=config.PADDLEOCR_REC_MODEL_DIR, print=False)
        else:
            self.ocr = PaddleOCR(det_db_unclip_ratio=2.2, use_gpu=True, use_angle_cls=True,  gpu_id=device, det_model_dir=config.PADDLEOCR_DETECTION_MODEL_DIR, cls_model_dir=config.PADDLEOCR_CLS_MODEL_DIR, rec_model_dir=config.PADDLEOCR_REC_MODEL_DIR, print=False)
            
        print('>> [PaddleOCR: DEVICE_ID = {0}] loaded model paddleOCR'.format(device), flush=True)

    def get_rotate_crop_image(self, img, points, padding_add, preprocess=False):
        # Use Green's theory to judge clockwise or counterclockwise
        if preprocess == True:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            img = cv2.filter2D(img, -1, kernel)

        d = 0.0
        for index in range(-1, 3):
            d += -0.5 * (points[index + 1][1] + points[index][1]) * (
                points[index + 1][0] - points[index][0])
        if d < 0:  # counterclockwise
            tmp = np.array(points)
            points[1], points[3] = tmp[3], tmp[1]

        try:
            img_crop_width = int(
                max(
                    np.linalg.norm(np.array(points[0]) - np.array(points[1])),
                    np.linalg.norm(np.array(points[2]) - np.array(points[3]))))
            img_crop_height = int(
                max(
                    np.linalg.norm(np.array(points[0]) - np.array(points[3])),
                    np.linalg.norm(np.array(points[1]) - np.array(points[2]))))
            
            # Padding 
            padding = int(min(img_crop_width, img_crop_height) * padding_add)

            pts_std = np.float32([[padding, padding], [img_crop_width + padding, padding],
                                [img_crop_width + padding, img_crop_height + padding],
                                [padding, img_crop_height + padding]])
            M = cv2.getPerspectiveTransform(np.float32(points), pts_std)
            dst_img = cv2.warpPerspective(
                img,
                M, (img_crop_width + padding*2, img_crop_height + padding*2),
                borderMode=cv2.BORDER_REPLICATE,
                flags=cv2.INTER_CUBIC)
            # dst_img_height, dst_img_width = dst_img.shape[0:2]
            # if dst_img_height * 1.0 / dst_img_width >= 1.5:
            #     dst_img = np.rot90(dst_img)
            return dst_img
        except Exception as e:
            print(e)

    def Detect_line(self, img):

        # if want post process by Noise Removal
        # img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)

        list_rotated_text = []
        result_boxes_list = self.ocr.ocr(img, det=True, rec=False, cls=True)[0]
        area_img_origin = img.shape[0] * img.shape[1]

        boxes_list = sorted(result_boxes_list, key=lambda box: box[0][1])

        # lọc trường number đầu
        # boxes_list = boxes_list[1:]

        for i, kps in enumerate(boxes_list):
            # area = ( max(point[0] for point in kps) -  min(point[0] for point in kps) ) * ( max(point[1] for point in kps) -  min(point[1] for point in kps) )
            rotated_text = self.get_rotate_crop_image(img, kps, 0.1)
            list_rotated_text.append(rotated_text)

        return list_rotated_text
    

    def Recognition(self, img, padding = False):
        
        # Preprocess
        # Get the aspect ratio of the image
        aspect_ratio = img.shape[1] / img.shape[0]

        # Calculate new width to maintain the same aspect ratio
        new_width = int(aspect_ratio * 100)

        # Resize the image to have a height of 32 while maintaining the aspect ratio

        if padding == True:
            resized_img = cv2.resize(img, (new_width, 100))
            # Get the amount of padding needed to reach a width of 480 pixels
            pad_left = (1500 - new_width) // 2
            pad_right = 1500 - new_width - pad_left

            if pad_left > 0:
                # Pad the resized image to reach a size of 32x480
                padded_img = cv2.copyMakeBorder(resized_img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[128, 128, 128])
            else:
                padded_img = cv2.resize(resized_img, (1500, 100))
        
        else:
            padded_img =  cv2.resize(img, (350, 200))

        result = self.ocr.ocr(padded_img, det=False, rec=True, cls=False)


        return result[0][0]

