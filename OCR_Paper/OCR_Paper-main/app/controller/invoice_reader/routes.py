# -*- coding: utf-8 -*-
import time
import uuid
from datetime import datetime
import socket
import cv2
from flask import Blueprint, request, jsonify
import numpy
import json
import dataclasses
from urllib.parse import urlparse

from app.extensions import logger

blueprint = Blueprint('invoice-information-extraction', __name__)
supported_types = ["ocr-reader"]
algorithms = dict()

def init():
    for name in supported_types:
        if name == "ocr-reader":
            from app.controller.invoice_reader import get_algorithm
            algorithm = get_algorithm()
            algorithms[name] = algorithm

@blueprint.route('/ocr-reader', methods=['POST'])
def read_sticker():
    # start_request = time.time()
    trace_id = str(uuid.uuid1())
    # fmt = '%Y-%m-%d %H:%M:%S'
    # now = datetime.now()

    try:
        img_file = request.files['image']
        filename = img_file.filename
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.HEIC', '.JPG', '.JPEG', '.PNG', '.heic', '.raw')):
            img_np_array = numpy.fromfile(img_file, dtype='uint8')
            img = cv2.imdecode(img_np_array, cv2.IMREAD_COLOR)
            # process img input
            results = algorithms["ocr-reader"].ocr_reader(img)

            # return result
            response_request = jsonify(success=True, data=results, trace_id=trace_id)

            return response_request, 200

        else:
            response_request = jsonify(success=False, message="cannot read file", trace_id=trace_id)
            return response_request, 400

    except Exception as e:
        response_request = jsonify(success=False, message="cannot validate : {0}".format(e), trace_id=trace_id)
        return response_request, 400
