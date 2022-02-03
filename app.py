import os
import io
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve
import json
import requests

from main import MaskDetection

app = Flask(__name__)
CORS(app)

DEVICE = os.environ['DEVICE'] if 'DEVICE' in os.environ else 'cpu'
FACE_DETECTION_URL = os.environ['FACE_DETECTION_URL'] if 'FACE_DETECTION_URL' in os.environ else 'http://192.168.5.5:8005'


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return jsonify(result='Success'), 200
    else:
        if 'file' not in request.files or not request.files['file']:
            return jsonify(error="file is required"), 422
        try:
            image_bytes = io.BytesIO(request.files['file'].read())
            img = Image.open(image_bytes)
        except IOError:
            return jsonify(error="invalid image"), 400

        image_bytes.seek(0)
        files = {'file': image_bytes}
        response = requests.post(FACE_DETECTION_URL, files=files)
        result = json.loads(response.text)["result"]
        if result:
            for face in result:
                face["_meta"] = {"image": img}

            detected_masks = mask_det(result)
            face_index = 0
            for face in result:
                del face["_meta"]
                face['mask'] = detected_masks[face_index]
                face_index += 1

        return jsonify(result=result), 200


if __name__ == '__main__':
    mask_det = MaskDetection(device=DEVICE)
    serve(app, host="0.0.0.0", port=8005)
