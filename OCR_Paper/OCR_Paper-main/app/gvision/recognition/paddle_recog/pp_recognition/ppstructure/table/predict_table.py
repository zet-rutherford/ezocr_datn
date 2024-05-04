# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import cv2
import copy
import logging
import numpy as np
import time
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.utility as utility
from tools.infer.predict_system import sorted_boxes_custom, sorted_boxes
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger
from ppstructure.table.matcher import TableMatch
from ppstructure.table.table_master_match import TableMasterMatcher
from ppstructure.utility import parse_args
import ppstructure.table.predict_structure as predict_strture

import numpy


logger = get_logger()

# TODO thainq: hợp nhất các bbox tên công ty với nhau để tránh việc bị miss một vài từ
def find_children_elem(it, list_elements, width_img, height_img):
    children = []
    threshold_y = height_img * 0.04
    threshold_x = width_img * 0.04
    
    width_current_elem = it['box'][2] - it['box'][0] #xmax-xmin
    for elem in list_elements:
        width_temp_elem = elem['box'][2] - elem['box'][0]

        if width_temp_elem > width_current_elem:
            longer_item = elem
            shorter_item = it
        else:
            longer_item = it
            shorter_item = elem

        width_longer_item = longer_item['box'][2] - longer_item['box'][0]
        width_shorter_item = shorter_item['box'][2] - shorter_item['box'][0]

        threshold_ratio_width = 0.45

        if abs(it['box'][1] - elem['box'][5]) <= threshold_y and abs(it['box'][0] - elem['box'][0]) <= threshold_x and \
                (width_shorter_item/width_longer_item) < threshold_ratio_width and longer_item['box'][1] < shorter_item['box'][1]:
            children.append(elem)
    return children

def merge_element(list_elements, width_img, height_img):
    list_line = []
    explored = []
    for elem in list_elements:
        line_current = []
        stack = []
        if elem not in explored:
            stack.append(elem)
        while len(stack) > 0:
            it = stack[-1]
            explored.append(it)
            stack.remove(it)
            if it not in line_current:
                line_current.append(it)

            children = find_children_elem(it, list_elements, width_img, height_img)
            for child in children:
                if child not in explored and children not in stack:
                    stack.append(child)
        if len(line_current) > 0:
            list_line.append(line_current)


    table_result = []
    for item in list_line:
        if len(item) > 1:
            item = sorted(item, key=lambda item: item['box'][1], reverse=False)
            table_item = {'text': ''}
            min_x = 99999
            min_y = 99999
            max_x = -1
            max_y = -1
            for subitem in item:
                table_item['score'] = subitem['score']
                table_item['text'] += subitem['text'] + ' '
                if subitem['box'][0] < min_x:
                    min_x = subitem['box'][0]
                if subitem['box'][1] < min_y:
                    min_y = subitem['box'][1]
                if subitem['box'][4] > max_x:
                    max_x = subitem['box'][4]
                if subitem['box'][5] > max_y:
                    max_y = subitem['box'][5]

            width_item = max_x - min_x
            height_item = max_y - min_y
            x1, y1, x2, y2, x3, y3, x4, y4 = min_x, min_y, min_x + width_item, min_y, min_x + width_item, min_y + height_item, min_x, min_y + height_item
            table_item['box'] = [x1, y1, x2, y2, x3, y3, x4, y4]
            table_result.append(table_item)
        elif len(item) == 1:
            table_result.append(item[0])

    return table_result

#########################

def expand(pix, det_box, shape):
    x0, y0, x1, y1 = det_box
    #     print(shape)
    h, w, c = shape
    tmp_x0 = x0 - pix
    tmp_x1 = x1 + pix
    tmp_y0 = y0 - pix
    tmp_y1 = y1 + pix
    x0_ = tmp_x0 if tmp_x0 >= 0 else 0
    x1_ = tmp_x1 if tmp_x1 <= w else w
    y0_ = tmp_y0 if tmp_y0 >= 0 else 0
    y1_ = tmp_y1 if tmp_y1 <= h else h
    return x0_, y0_, x1_, y1_


class TableSystem(object):
    def __init__(self, args, text_detector=None, text_recognizer=None):
        self.args = args
        if not args.show_log:
            logger.setLevel(logging.INFO)
        benchmark_tmp = False
        if args.benchmark:
            benchmark_tmp = args.benchmark
            args.benchmark = False
        self.text_detector = predict_det.TextDetector(copy.deepcopy(
            args)) if text_detector is None else text_detector
        self.text_recognizer = predict_rec.TextRecognizer(copy.deepcopy(
            args)) if text_recognizer is None else text_recognizer
        if benchmark_tmp:
            args.benchmark = True
        self.table_structurer = predict_strture.TableStructurer(args)
        if args.table_algorithm in ['TableMaster']:
            print('1')
            self.match = TableMasterMatcher()
        else:
            print('2')
            self.match = TableMatch(filter_ocr_result=True)

        self.predictor, self.input_tensor, self.output_tensors, self.config = utility.create_predictor(
            args, 'table', logger)

    def __call__(self, img, return_ocr_result_in_table=False, custom_text_detector=None, custome_text_recognizer=None):
        result = dict()
        time_dict = {'det': 0, 'rec': 0, 'table': 0, 'all': 0, 'match': 0}
        start = time.time()
        structure_res, elapse = self._structure(copy.deepcopy(img))
        result['cell_bbox'] = structure_res[1].tolist()
        time_dict['table'] = elapse

        # TODO thainq: comment code
        # dt_boxes, rec_res, det_elapse, rec_elapse = self._ocr(
        #     copy.deepcopy(img))
        time_dict['det'] = 0
        time_dict['rec'] = 0



        """
        TODO thainq: text detection and text recognition
        """ 
        dt_boxes = []
        rec_res = []

        list_boxes = []
        list_crop_imgs = []
        list_elements = []
        
        text_detected = custom_text_detector.detect(img)
        for item_dt in text_detected:
            x1, y1, x2, y2, x3, y3, x4, y4 = item_dt['box']
            if x3-x1 < 10 or y3-y1<10:
                continue
            list_boxes.append(item_dt['box'])
            list_crop_imgs.append(img[y1:y3, x1:x3])

        pred_text, pred_score = custome_text_recognizer.recog(list_crop_imgs)
        for i in range(len(pred_text)):
            elem = {}
            elem['box'] = list_boxes[i]
            elem['text'] = pred_text[i]
            elem['score'] = pred_score[i]
            list_elements.append(elem)
        

        list_elements_merged = merge_element(list_elements, img.shape[1], img.shape[0])
        sorted_list = sorted_boxes_custom(list_elements_merged)

        for sorted_item in sorted_list:
            x1, y1, x2, y2, x3, y3, x4, y4 = sorted_item['box']
            content = sorted_item['text']
            confidence = sorted_item['score']

            item_dt = [x1, y1, x3, y3]
            dt_boxes.append(item_dt)
            rec_res.append(tuple([content, confidence]))
        
        dt_boxes = numpy.asarray(dt_boxes)
        ###################################################################


        if return_ocr_result_in_table:
            print('return_ocr_result_in_table')
            result['boxes'] = dt_boxes  #[x.tolist() for x in dt_boxes]
            result['rec_res'] = rec_res

        tic = time.time()
        pred_html = self.match(structure_res, dt_boxes, rec_res)
        toc = time.time()
        time_dict['match'] = toc - tic
        result['html'] = pred_html
        end = time.time()
        time_dict['all'] = end - start
        return result, time_dict

    def _structure(self, img):
        structure_res, elapse = self.table_structurer(copy.deepcopy(img))
        return structure_res, elapse

    def _ocr(self, img):
        h, w = img.shape[:2]
        dt_boxes, det_elapse = self.text_detector(copy.deepcopy(img))
        dt_boxes = sorted_boxes(dt_boxes)

        r_boxes = []
        for box in dt_boxes:
            x_min = max(0, box[:, 0].min() - 1)
            x_max = min(w, box[:, 0].max() + 1)
            y_min = max(0, box[:, 1].min() - 1)
            y_max = min(h, box[:, 1].max() + 1)
            box = [x_min, y_min, x_max, y_max]
            r_boxes.append(box)
        dt_boxes = np.array(r_boxes)
        logger.debug("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), det_elapse))
        if dt_boxes is None:
            return None, None

        img_crop_list = []
        for i in range(len(dt_boxes)):
            det_box = dt_boxes[i]
            x0, y0, x1, y1 = expand(2, det_box, img.shape)
            text_rect = img[int(y0):int(y1), int(x0):int(x1), :]
            img_crop_list.append(text_rect)
        rec_res, rec_elapse = self.text_recognizer(img_crop_list)
        logger.debug("rec_res num  : {}, elapse : {}".format(
            len(rec_res), rec_elapse))
        return dt_boxes, rec_res, det_elapse, rec_elapse


def to_excel(html_table, excel_path):
    from tablepyxl import tablepyxl
    tablepyxl.document_to_xl(html_table, excel_path)


from premailer import Premailer
def get_table_structure(html_table):
    from tablepyxl import tablepyxl
    inline_styles_doc = Premailer(html_table, base_url=None, remove_classes=False).transform()
    tables = tablepyxl.get_Tables(inline_styles_doc)
    return tables



def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list[args.process_id::args.total_process_num]
    os.makedirs(args.output, exist_ok=True)

    table_sys = TableSystem(args)
    img_num = len(image_file_list)

    f_html = open(
        os.path.join(args.output, 'show.html'), mode='w', encoding='utf-8')
    f_html.write('<html>\n<body>\n')
    f_html.write('<table border="1">\n')
    f_html.write(
        "<meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\" />"
    )
    f_html.write("<tr>\n")
    f_html.write('<td>img name\n')
    f_html.write('<td>ori image</td>')
    f_html.write('<td>table html</td>')
    f_html.write('<td>cell box</td>')
    f_html.write("</tr>\n")

    for i, image_file in enumerate(image_file_list):
        logger.info("[{}/{}] {}".format(i, img_num, image_file))
        img, flag, _ = check_and_read(image_file)
        excel_path = os.path.join(
            args.output, os.path.basename(image_file).split('.')[0] + '.xlsx')
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.error("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        pred_res, _ = table_sys(img)
        pred_html = pred_res['html']
        logger.info(pred_html)
        to_excel(pred_html, excel_path)
        logger.info('excel saved to {}'.format(excel_path))
        elapse = time.time() - starttime
        logger.info("Predict time : {:.3f}s".format(elapse))

        if len(pred_res['cell_bbox']) > 0 and len(pred_res['cell_bbox'][
                0]) == 4:
            img = predict_strture.draw_rectangle(image_file,
                                                 pred_res['cell_bbox'])
        else:
            img = utility.draw_boxes(img, pred_res['cell_bbox'])
        img_save_path = os.path.join(args.output, os.path.basename(image_file))
        cv2.imwrite(img_save_path, img)

        f_html.write("<tr>\n")
        f_html.write(f'<td> {os.path.basename(image_file)} <br/>\n')
        f_html.write(f'<td><img src="{image_file}" width=640></td>\n')
        f_html.write('<td><table  border="1">' + pred_html.replace(
            '<html><body><table>', '').replace('</table></body></html>', '') +
                     '</table></td>\n')
        f_html.write(
            f'<td><img src="{os.path.basename(image_file)}" width=640></td>\n')
        f_html.write("</tr>\n")
    f_html.write("</table>\n")
    f_html.close()

    if args.benchmark:
        table_sys.table_structurer.autolog.report()


if __name__ == "__main__":
    args = parse_args()
    if args.use_mp:
        import subprocess
        p_list = []
        total_process_num = args.total_process_num
        for process_id in range(total_process_num):
            cmd = [sys.executable, "-u"] + sys.argv + [
                "--process_id={}".format(process_id),
                "--use_mp={}".format(False)
            ]
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
            p_list.append(p)
        for p in p_list:
            p.wait()
    else:
        main(args)
