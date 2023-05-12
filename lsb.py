import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class LSB:
    def binarize(self, image_to_transform, threshold):
        # Converting to grayscale
        output_image = image_to_transform.convert("L")
        for x in range(output_image.width):
            for y in range(output_image.height):
                if output_image.getpixel((x, y)) < threshold:
                    output_image.putpixel((x, y), 0)
                else:
                    output_image.putpixel((x, y), 255)
        return output_image

    def watermark(self, img, watermark):
        r, g, b = img.split()

        # Flatten both images and convert it into numpy array
        array = np.array(list(r.getdata()))
        array1 = np.array(list(watermark.getdata()))

        # Watermarking original image by appending 0 or 1 to the original image according to the pixel intensity of the watermarking image
        for p in range(img.size[0]*img.size[1]):
            bin1 = bin(array[p])[2:-1]
            x = bin(array1[p])[2]
            bin1 += x
            array[p] = int(bin1, 2)

        # Converting watermarked numpy array back to image
        array = array.reshape(1000, 1000)
        red = Image.fromarray(array.astype('uint8'), 'L')
        watermarked_image = Image.merge('RGB', (red, g, b))
        return watermarked_image

    def extract(self, watermarked_image):
        array2 = np.array(list(watermarked_image.split()[0].getdata()))
        for p in range(1000000):
            if bin(array2[p])[-1] == '1':
                array2[p] = 255
            else:
                array2[p] = 0

        # Display extracted watermark
        array2 = array2.reshape(1000, 1000)
        extract_watermark = Image.fromarray(array2.astype('uint8'), 'L')
        return extract_watermark


def LSBwatermarking():
    lsb = LSB()
    plt.rcParams['figure.figsize'] = [15, 15]

    # Image to be watermarked
    img = Image.open('input/LSBimage.jpg')
    # Resize image
    newsize = (1000, 1000)
    img = img.resize(newsize)

    # Watermarking Image
    watermark = Image.open('input/LSBwatermark.jpg')
    # Resize image
    newsize = (1000, 1000)
    watermark = watermark.resize(newsize)
    watermark = lsb.binarize(watermark, 128)

    watermarked_img = lsb.watermark(img, watermark)

    extract_watermark = lsb.extract(watermarked_img)
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.title("original")
    plt.subplot(2, 2, 2)
    plt.imshow(watermarked_img, cmap='gray')
    plt.axis("off")
    plt.title("watermarked image")
    plt.subplot(2, 2, 3)
    plt.imshow(watermark, cmap='gray')
    plt.title("original")
    plt.subplot(2, 2, 4)
    plt.imshow(extract_watermark, cmap='gray')
    plt.title("extracted watermark")
    plt.show()

    watermarked_img = watermarked_img.save("output/LSBimage_output.jpg")
    extract_watermark = extract_watermark.save(
        "output/LSBwatermark_output.jpg")


if __name__ == '__main__':
    LSBwatermarking()
