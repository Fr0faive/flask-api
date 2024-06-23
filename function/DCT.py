# DCT.py

import cv2
import numpy as np

def dct2(block):
    return cv2.dct(block.astype(np.float32))

def idct2(block):
    return cv2.idct(block.astype(np.float32))

def compress_frame(frame, block_size=8, quantization_factor=0.1, retain_percentage=0.1):
    h, w = frame.shape[:2]
    compressed_frame = np.zeros_like(frame, dtype=np.uint8)

    for channel in range(frame.shape[2]):
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = frame[i:i+block_size, j:j+block_size, channel]
                if block.shape[0] < block_size or block.shape[1] < block_size:
                    padded_block = np.zeros((block_size, block_size), dtype=np.float32)
                    padded_block[:block.shape[0], :block.shape[1]] = block
                    block = padded_block
                dct_block = dct2(block)
                quantized_block = np.round(dct_block * quantization_factor)

                flat_quantized = quantized_block.flatten()
                num_coeffs = int(np.prod(block.shape) * retain_percentage)
                sorted_coeffs = np.argsort(np.abs(flat_quantized))
                quantized_block.flatten()[sorted_coeffs[:-num_coeffs]] = 0

                idct_block = idct2(quantized_block / quantization_factor)
                idct_block_resized = cv2.resize(idct_block, (block_size, block_size))

                compressed_frame[i:i+block_size, j:j+block_size, channel] = idct_block_resized

    return compressed_frame

def compress_video(input_file, output_file, block_size=8, quantization_factor=0.1, scale_factor=0.5):
    cap = cv2.VideoCapture(input_file)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_factor)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_factor)
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        compressed_frame = compress_frame(frame, block_size=block_size, quantization_factor=quantization_factor, retain_percentage=0.05)
        out.write(compressed_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
