import cv2
import numpy as np
import pywt

def dwt2(block):
    coeffs = pywt.dwt2(block, 'haar')
    cA, (cH, cV, cD) = coeffs
    return cA, cH, cV, cD

def idwt2(cA, cH, cV, cD):
    coeffs = (cA, (cH, cV, cD))
    return pywt.idwt2(coeffs, 'haar')

def compress_frame_dwt(frame, block_size=8, quantization_factor=0.1, retain_percentage=0.1):
    h, w = frame.shape[:2]
    compressed_frame = np.zeros_like(frame, dtype=np.uint8)

    for channel in range(frame.shape[2]):
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = frame[i:i+block_size, j:j+block_size, channel]
                if block.shape[0] < block_size or block.shape[1] < block_size:
                    # Handle edge case where block size is smaller than expected
                    padded_block = np.zeros((block_size, block_size), dtype=np.float32)
                    padded_block[:block.shape[0], :block.shape[1]] = block
                    block = padded_block

                cA, cH, cV, cD = dwt2(block)
                cA_quantized = np.round(cA * quantization_factor)
                cH_quantized = np.round(cH * quantization_factor)
                cV_quantized = np.round(cV * quantization_factor)
                cD_quantized = np.round(cD * quantization_factor)

                # Retain only a certain percentage of the most significant coefficients
                def retain_significant_coeffs(coeffs, retain_percentage):
                    flat_coeffs = coeffs.flatten()
                    num_coeffs = int(len(flat_coeffs) * retain_percentage)
                    sorted_coeffs = np.argsort(np.abs(flat_coeffs))
                    flat_coeffs[sorted_coeffs[:-num_coeffs]] = 0
                    return flat_coeffs.reshape(coeffs.shape)

                cA_quantized = retain_significant_coeffs(cA_quantized, retain_percentage)
                cH_quantized = retain_significant_coeffs(cH_quantized, retain_percentage)
                cV_quantized = retain_significant_coeffs(cV_quantized, retain_percentage)
                cD_quantized = retain_significant_coeffs(cD_quantized, retain_percentage)

                idwt_block = idwt2(cA_quantized / quantization_factor, cH_quantized / quantization_factor, cV_quantized / quantization_factor, cD_quantized / quantization_factor)

                # Resize idwt_block to match the original block size
                idwt_block_resized = cv2.resize(idwt_block, (block_size, block_size))

                compressed_frame[i:i+block_size, j:j+block_size, channel] = idwt_block_resized

    return compressed_frame

def compress_video_dwt(input_file, output_file, block_size=8, quantization_factor=0.1, scale_factor=0.5):
    cap = cv2.VideoCapture(input_file)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_factor)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_factor)
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_count)
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        print(frame_idx)
        if not ret:
            break

        # Resize frame
        frame = cv2.resize(frame, (width, height))

        compressed_frame = compress_frame_dwt(frame, block_size=block_size, quantization_factor=quantization_factor, retain_percentage=0.05)
        out.write(compressed_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
