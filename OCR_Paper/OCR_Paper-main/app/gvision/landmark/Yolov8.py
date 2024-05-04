import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch

def get_rotate_crop_image(img, points, pad):
    # Use Green's theory to judge clockwise or counterclockwise
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
        padding = int(min(img_crop_width, img_crop_height) * pad)

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



def crop_image(image, box):
    [xmin, ymin, xmax, ymax ]= box
    cropped_image = image[int(ymin):int(ymax), int(xmin):int(xmax)]
    return cropped_image

def crop_image_with_padding(image, box, padding_percentage = 0.2):
    xmin, ymin, xmax, ymax = box

    # Calculate the width and height of the bounding box
    width = xmax - xmin
    height = ymax - ymin

    # Calculate the padding values
    padding_width = int(width * padding_percentage)
    padding_height = int(height * padding_percentage)

    # Add padding to the bounding box
    xmin -= padding_width
    ymin -= padding_height
    xmax += padding_width
    ymax += padding_height

    # Ensure that the bounding box coordinates are within the image boundaries
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(image.shape[1], xmax)
    ymax = min(image.shape[0], ymax)

    # Crop the image with padding
    cropped_image = image[ymin:ymax, xmin:xmax].copy()

    return cropped_image

def check_box(keypoints):
    [keypoint1, keypoint2, keypoint3, keypoint4] = keypoints

    # Tính vector từ keypoint1 đến keypoint2
    vector1 = [keypoint2[0] - keypoint1[0], keypoint2[1] - keypoint1[1]]
    # Tính vector từ keypoint2 đến keypoint3
    vector2 = [keypoint3[0] - keypoint2[0], keypoint3[1] - keypoint2[1]]
    # Tính vector từ keypoint3 đến keypoint4
    vector3 = [keypoint4[0] - keypoint3[0], keypoint4[1] - keypoint3[1]]
    # Tính vector từ keypoint4 đến keypoint1
    vector4 = [keypoint1[0] - keypoint4[0], keypoint1[1] - keypoint4[1]]

    # Tính tích vô hướng của hai vector liên tiếp
    cross_product1 = vector1[0] * vector2[1] - vector1[1] * vector2[0]
    cross_product2 = vector2[0] * vector3[1] - vector2[1] * vector3[0]
    cross_product3 = vector3[0] * vector4[1] - vector3[1] * vector4[0]

    # Kiểm tra tích vô hướng
    if cross_product1 * cross_product2 > 0 and cross_product2 * cross_product3 > 0:
        return True
    else:
        return False
    
def calculate_iou(box1, box2):
    # box1 and box2 are lists of [x1, y1, x2, y2] coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    if intersection == 0:
        iou = 0
    else:
        iou = intersection / float(area1 + area2 - intersection)
    return iou

def convert_points2boxxy(points):
    box_x = []
    box_y = []
    for box in points:
        box_x.append(box[0])
        box_y.append(box[1])
    return [min(box_x), min(box_y), max(box_x), max(box_y)]

def find_box(points, box_list):
    box_points = convert_points2boxxy(points)
    
    max_iou = 0
    for j, box in enumerate(box_list):
        iou = calculate_iou(box_points, box)
        if iou > max_iou:
            max_iou = iou
    return j


class Yolov8_Landmark:
    def __init__(self, model_dir: Path, device: int, image_size: int = 640, conf: float = 0.7, iou: float = 0.7) -> None:
        """
        instanciate the model.

        Parameters
        ----------
        model_dir : Path
            directory where to find the model weights.

        device : str
            the device name to run the model on.
        """
        self.model = YOLO(model_dir)
        self.image_size = image_size
        self.conf = conf
        self.iou = iou
        if device < 0:
            self.device = 'cpu'
        else:
            self.device = 'cuda:{0}'.format(device)
        
        print('[YOLOv8-landmark: DEVICE_ID = {0}] loaded model SHOP_ID = {1}'.format(device, model_dir))


    def get_model_predict(self, input_image, padding = 0):
        """
        Get the predictions of a model on an input image.

        Args:
            model (YOLO): The trained YOLO model.
            input_image (Image): The image on which the model will make predictions.

        Returns:
            pd.DataFrame: A DataFrame containing the predictions.
        """
        # Make predictions
        predictions = self.model.predict(
                            imgsz=self.image_size,
                            source=input_image,
                            conf=self.conf,
                            iou=self.iou,
                            device=self.device,
                            verbose=False
                            )
        if self.device != 'cpu':
            print('current device : ', torch.cuda.current_device())
        
        keypoints = predictions[0].to("cpu").numpy().keypoints.xy
        result_cropeds = []
        list_boxes = []
        for i, box in enumerate(predictions[0].boxes):
            new_box = [int(num) for num in box.xyxy[0]]
            list_boxes.append(new_box)
            # image croper
            image_croper = crop_image_with_padding( input_image, new_box, padding)
            result_cropeds.append(image_croper)
        
        result_warpeds = []
        for i, keypoint in enumerate(keypoints):
            # image warper
            if len(keypoint) > 0:
                if len(keypoint) == 4:
                    contour = np.array(keypoint, dtype=np.int32)
                    is_convex = cv2.isContourConvex(contour)
                    # is_convex = False
                    if is_convex:
                        image_warper = get_rotate_crop_image(input_image, keypoint, padding)
                    else:
                        image_warper = result_cropeds[find_box(keypoint, list_boxes)]
                
                else:
                    image_warper = result_cropeds[find_box(keypoint, list_boxes)]

                result_warpeds.append(image_warper)

        return result_cropeds, result_warpeds, keypoints