import cv2
import numpy as np
import pywt


def embed_watermark(cover_img_path, watermark_img_path, output_path):
    # Load images
    cover_img = cv2.imread(cover_img_path)
    watermark_img = cv2.imread(watermark_img_path)

    # Resize watermark image to match the size of the cover image
    
    watermark_img_resized = cv2.resize(watermark_img, (cover_img.shape[1], cover_img.shape[0]))

    # Convert images to float
    cover_img_float = cover_img.astype(np.float32) / 255.0
    watermark_img_float = watermark_img_resized.astype(np.float32) / 255.0

    # Apply DWT to cover and watermark images
    cover_coeffs = pywt.dwt2(cover_img_float, "haar")
    watermark_coeffs = pywt.dwt2(watermark_img_float, "haar")

    # Embed watermark in the LL subband of cover image
    watermarked_coeffs = list(cover_coeffs)
    watermarked_coeffs[0] += (0.1 * watermark_coeffs[0])

    # Reconstruct watermarked image
    watermarked_img_float = pywt.idwt2(watermarked_coeffs, "haar")
    watermarked_img = (watermarked_img_float * 255).astype(np.uint8)

    # Save watermarked image
    cv2.imwrite(output_path, watermarked_img)
    print("Watermark embedded successfully.")


def extract_watermark(watermarked_img_path, output_path):
    # Load watermarked image
    watermarked_img = cv2.imread(watermarked_img_path)

    # Convert image to grayscale
    watermarked_gray = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2GRAY)

    # Apply DWT to watermarked image
    watermarked_coeffs = pywt.dwt2(watermarked_gray, "haar")

    # Extract watermark from the LL subband of watermarked image
    watermark_coeffs = list(watermarked_coeffs)
    watermark_coeffs[0] /= 0.1
    watermark_gray = pywt.idwt2(watermark_coeffs, "haar")

    # Convert grayscale watermark to RGB
    watermark_img = cv2.cvtColor(watermark_gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Save extracted watermark
    cv2.imwrite(output_path, watermark_img)
    print("Watermark extracted successfully.")


# ==================================

# Example usage:

cover_img_path = "input\DWTcover.jpg"
watermark_img_path = "input\DCTapple-watermark.png"
watermarked_img_path = "output\DWTwatermarked.jpg"
extracted_watermark_path = "output\DWTextracted_watermark.png"

embed_watermark(cover_img_path, watermark_img_path, watermarked_img_path)
extract_watermark(watermarked_img_path, extracted_watermark_path)