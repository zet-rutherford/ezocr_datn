import re
import cv2
def postprocess_name_prd(list_string):
    result_string = ' '.join(list_string)
    digits_num_list = re.findall(r'\d+', result_string)
    
    # Initialize the contiguous character sequences with the entire result_string
    contiguous_char_sequences_list = [result_string]
    
    for digits_num in digits_num_list:
        if len(digits_num) > 8:
            # If a sequence of digits has length > 8, remove it from the list of sequences
            contiguous_char_sequences_list = [sequence.replace(digits_num, '|') for sequence in contiguous_char_sequences_list]
    
    return contiguous_char_sequences_list[0]

class Algorithm():
    def __init__(self, landmark_detector, paddleocr_text_detector, vietocr_recognizer):
        self.landmark_detector = landmark_detector
        self.paddleocr_text_detector = paddleocr_text_detector
        self.vietocr_recognizer = vietocr_recognizer
        
    def ocr_reader(self, org_img):
        #   input is 1 img ===> output is text4
        #   Detect line 
        result_rotated_texts = self.paddleocr_text_detector.Detect_line(org_img)

        #   OCR
        result_text_recog_list, _ = self.vietocr_recognizer.recog(result_rotated_texts)
        result_text_recog_list_return = []
        for result_text_recog in result_text_recog_list:
            if len(result_text_recog) > 2:
                result_text_recog_list_return.append(result_text_recog)


        return result_text_recog_list_return