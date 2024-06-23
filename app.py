from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
import tempfile
from function.DCT import compress_video
from function.DWT import compress_video_dwt

app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
  return "Hello World, Server Working!"

@app.route('/dct-compressor', methods=['POST'])
def dct_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    with tempfile.NamedTemporaryFile(delete=False) as input_tempfile:
        file.save(input_tempfile.name)
        output_tempfile = tempfile.NamedTemporaryFile(delete=False)
        try:
            compress_video(input_tempfile.name, output_tempfile.name, block_size=8, quantization_factor=8, scale_factor=0.2)
        finally:
            input_tempfile.close()

        return send_file(output_tempfile.name, as_attachment=True, attachment_filename='compressed_video.mp4')

@app.route('/dwt-compressor', methods=['POST'])
def dwt_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    with tempfile.NamedTemporaryFile(delete=False) as input_tempfile:
        file.save(input_tempfile.name)
        output_tempfile = tempfile.NamedTemporaryFile(delete=False)
        try:
            compress_video_dwt(input_tempfile.name, output_tempfile.name, block_size=8, quantization_factor=8, scale_factor=0.2)
        finally:
            input_tempfile.close()

        return send_file(output_tempfile.name, as_attachment=True, attachment_filename='compressed_video.mp4')

if __name__ == '__main__':
    app.run(debug=False)  # Set debug=True for debugging purposes
