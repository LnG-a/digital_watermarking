import cv2
import numpy as np
import random

# Đọc ảnh gốc và watermark
img_path = 'input\DFTimage.jpeg'
watermark_path = 'input\DFTwatermark.jpeg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

# Lấy kích thước ảnh và thêm padding cho ảnh watermark
img_height, img_width = img.shape
watermark_height, watermark_width = watermark.shape
pad_height = img_height - watermark_height
pad_width = img_width - watermark_width
watermark_padded = cv2.copyMakeBorder(watermark, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=0)

# Áp dụng phép biến đổi DFT lên ảnh và watermark
img_dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
watermark_dft = cv2.dft(np.float32(watermark_padded), flags=cv2.DFT_COMPLEX_OUTPUT)

# Chọn ngẫu nhiên các tần số để chèn watermark
num_coeffs = 1000
max_height, max_width = img_height//2, img_width//2
coords = [(random.randint(0, max_height), random.randint(0, max_width)) for _ in range(num_coeffs)]

# Thêm watermark vào các tần số được chọn
for coord in coords:
    i, j = coord
    img_dft[i, j] = watermark_dft[i, j]


# Áp dụng phép biến đổi ngược (IDFT) để lấy ảnh đã được chèn watermark
img_watermarked = cv2.idft(img_dft, flags=cv2.DFT_COMPLEX_OUTPUT)
img_watermarked = cv2.magnitude(img_watermarked[:, :, 0], img_watermarked[:, :, 1])
img_watermarked = cv2.normalize(img_watermarked, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Lưu ảnh ra đã được chèn watermark ra file
cv2.imwrite('output\DFTwatermarked.jpg', img_watermarked)

# Trích xuất watermark

# Áp dụng phép biến đổi DFT lên ảnh đã được chèn watermark
dft = cv2.dft(np.float32(img_watermarked), flags=cv2.DFT_COMPLEX_OUTPUT)

# Trích xuất các giá trị tại các tần số đã chọn từ ma trận phức đã được tính toán
watermark_dft = np.zeros((img_height, img_width, 2), dtype=np.float32)
for coord in coords:
    i, j = coord
    watermark_dft[i, j] = dft[i, j]

# Áp dụng phép biến đổi ngược (IDFT) lên ma trận tần số đã trích xuất để lấy watermark
watermark_idft = cv2.idft(watermark_dft, flags=cv2.DFT_COMPLEX_OUTPUT)
watermark_idft = cv2.magnitude(watermark_idft[:, :, 0], watermark_idft[:, :, 1])
watermark_idft = cv2.normalize(watermark_idft, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Lưu ảnh ra đã trích xuất watermark ra file
cv2.imwrite('output\DFT_extracted_watermark.jpg', watermark_idft)

# Hiển thị watermark đã được trích xuất
watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
cv2.imshow('Watermark', watermark)
cv2.imshow('Extracted Watermark', watermark_idft)
cv2.waitKey(0)
cv2.destroyAllWindows()

