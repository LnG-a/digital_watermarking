import cv2
import numpy as np
import pywt

def embed_watermark(image_path, watermark_path, output_path, secret_key):
    # Load the image
    image = cv2.imread(image_path)

    # Resize the image to a multiple of 32
    resized_image = resize_image(image)

    # Convert the resized image to YCbCr color space
    ycrcb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YCrCb)

    # Extract the Y channel
    y_channel = ycrcb_image[:, :, 0]

    # Apply DWT to the Y channel
    coeffs = pywt.dwt2(y_channel, 'haar')
    LL, (LH, HL, HH) = coeffs

    # Apply DWT to LH component
    coeffs_LH = pywt.dwt2(LH, 'haar')
    LL2, (LH2, HL2, HH2) = coeffs_LH

    # Get the watermark image
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    watermark = resize_image(watermark)

    # Convert the watermark to binary message vector (MV)
    message_vector = watermark.flatten() > 127

    # Generate pseudo-noise sequence (PSN) using secret key
    np.random.seed(secret_key)
    psn = np.random.randint(0, 256, size=len(message_vector), dtype=np.uint8)

    # Embed the watermark into LH2 coefficients
    embedded_coeffs = embed_watermark_coeffs(LH2, message_vector, psn)

    # Reconstruct LH component using inverse DWT
    reconstructed_LH = pywt.idwt2((LL2, (embedded_coeffs, HL2, HH2)), 'haar')

    # Reconstruct the Y channel using inverse DWT
    reconstructed_y_channel = pywt.idwt2((LL, (reconstructed_LH, HL, HH)), 'haar')

    # Replace the Y channel in the YCrCb image
    ycrcb_image[:, :, 0] = reconstructed_y_channel

    # Convert the YCrCb image back to BGR color space
    reconstructed_image = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2BGR)

    # Save the watermarked image
    cv2.imwrite(output_path, reconstructed_image)
    print("Watermark embedded successfully.")


def resize_image(image, size_multiple=32):
    height, width = image.shape[:2]
    resized_height = (height // size_multiple) * size_multiple
    resized_width = (width // size_multiple) * size_multiple
    resized_image = cv2.resize(image, (resized_width, resized_height))
    return resized_image


def embed_watermark_coeffs(coeffs, message_vector, psn):
    embedded_coeffs = np.copy(coeffs)
    index = 0
    for i in range(len(coeffs)):
        block = coeffs[i]
        if block.shape[0] == 8 and block.shape[1] == 8:
            if message_vector[index] == 0:
                embedded_coeffs[i] = block + psn[index]
            index += 1
    return embedded_coeffs

def extract_watermark(embedded_img_path, output_path):
    # Load the embedded image
    embedded_img = cv2.imread(embedded_img_path)

    # Convert the image to YCbCr color space
    embedded_img_ycrcb = cv2.cvtColor(embedded_img, cv2.COLOR_BGR2YCrCb)

    # Extract the Y channel (luminance component)
    embedded_y_channel = embedded_img_ycrcb[:, :, 0]

    # Apply DWT to the Y channel
    coeffs = pywt.dwt2(embedded_y_channel, "haar")

    # Extract watermark from the LH subband
    lh_subband = coeffs[1][1]  # Extract the LH subband
    watermark = np.zeros_like(lh_subband)

    # Extract the pseudo-noise sequence (PSN)
    psn = np.random.RandomState(123).randn(lh_subband.shape[0], lh_subband.shape[1])

    # Extract the watermark from the PSN and watermark coefficients
    for i in range(lh_subband.shape[0]):
        for j in range(lh_subband.shape[1]):
            if lh_subband[i, j] > 0:
                watermark[i, j] = 1 if lh_subband[i, j] > psn[i, j] else 0

    # Save the extracted watermark as an image
    cv2.imwrite(output_path, watermark * 255)
    print("Watermark extracted successfully.")

# Example usage:

cover_img_path = "input\DWTcover.jpg"
watermark_img_path = "input\DCTapple-watermark.png"
watermarked_img_path = "output\DWTwatermarked.jpg"
extracted_watermark_path = "output\DWTextracted_watermark.png"
secret_key = 12345

# embed_watermark(cover_img_path, watermark_img_path, watermarked_img_path, secret_key)
# extract_watermark(watermarked_img_path, extracted_watermark_path)

import tkinter as tk
from tkinter import filedialog

def embed_wm():
    cover_img_path = filedialog.askopenfilename(title="Select Cover Image")
    watermark_img_path = filedialog.askopenfilename(title="Select Watermark Image")
    output_path = filedialog.asksaveasfilename(title="Save Watermarked Image")
    
    # Tiếp tục thực hiện các bước nhúng watermark
    embed_watermark(cover_img_path, watermark_img_path, output_path, secret_key)
    # Gọi hàm embed_watermark với các tham số đã lựa chọn
    
def extract_wm():
    watermarked_img_path = filedialog.askopenfilename(title="Select Watermarked Image")
    output_path = filedialog.asksaveasfilename(title="Save Extracted Watermark")
    
    # Tiếp tục thực hiện các bước trích xuất watermark
    extract_watermark(watermarked_img_path, output_path)
    
    # Gọi hàm extract_watermark với các tham số đã lựa chọn

# Tạo cửa sổ giao diện
window = tk.Tk()
window.title("Watermark Embedding and Extraction")

# Tạo các nút và khung chứa
embed_button = tk.Button(window, text="Embed Watermark", command=embed_wm)
embed_button.pack()

extract_button = tk.Button(window, text="Extract Watermark", command=extract_wm)
extract_button.pack()

# Chạy vòng lặp chờ các sự kiện của giao diện
window.mainloop()
