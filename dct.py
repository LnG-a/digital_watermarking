import cv2
import numpy as np
import matplotlib.pyplot as plt


class DCTWatermark:
    @staticmethod
    def generate_signature(wm, size):
        wm = cv2.resize(wm, (size, size))
        wm = np.where(wm < np.mean(wm), 0, 1)
        return wm

    def embed(self, cover, wm):
        Q = 10
        size = 2
        sig_size = 100

        img = cv2.cvtColor(cover, cv2.COLOR_BGR2YUV)
        signature = self.generate_signature(wm, sig_size).flatten()
        Y = img[:, :, 0]    # Get the first channel of img

        # Watermark góc thứ nhất của ảnh
        for x in range(0, sig_size * size, size):
            for y in range(0, sig_size * size, size):

                v = np.float32(Y[x:x + size, y:y + size])   # Get block thứ nhất
                v = cv2.dct(v)                              # dct block thứ nhất
                v[size - 1, size - 1] = Q * \
                    signature[(x // size) * sig_size + y // size]  # Thay đổi giá trị dct ở điểm cuối
                v = cv2.idct(v)                             # idct block thứ nhất

                # adjust v value to statisfy [0, 255]
                maximum = max(v.flatten())
                minimum = min(v.flatten())
                if maximum > 255:
                    v = v - (maximum - 255)
                if minimum < 0:
                    v = v - minimum

                Y[x:x + size, y:y + size] = v               # Cập nhật lại block thứ nhất của ảnh

        img[:, :, 0] = Y
        cover = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)

        return cover

    @staticmethod
    def extract(wmimg):
        Q = 10
        size = 2
        sig_size = 100

        (Y, U, V) = cv2.split(cv2.cvtColor(wmimg, cv2.COLOR_BGR2YUV))

        ext_sig = np.zeros(sig_size ** 2)

        for x in range(0, sig_size * size, size):
            for y in range(0, sig_size * size, size):
                v = cv2.dct(np.float32(Y[x:x + size, y:y + size]))
                if v[size - 1, size - 1] > Q / 2:
                    ext_sig[(x // size) * sig_size + y // size] = 1

        result = [ext_sig][0]
        result = np.array(result).reshape((sig_size, sig_size))
        result = np.where(result == 1, 255, 0)
        return result


if __name__ == "__main__":
    img = cv2.imread("images/data/food.jpg")
    wm = cv2.imread("images/watermark/apple-watermark.png", cv2.IMREAD_GRAYSCALE)
    dct = DCTWatermark()
    wmd = dct.embed(img, wm)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(10, 5))
    # Plot the first image on ax1
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original')

    # Plot the second imag  e on ax2
    ax2.imshow(cv2.resize(wm, (100, 100)))
    ax2.set_title('Watermark')

    # Plot the third image on ax3
    ax3.imshow(cv2.cvtColor(wmd, cv2.COLOR_BGR2RGB))
    ax3.set_title('Watermarked')

    signature = dct.extract(wmd)

    # Plot the third image on ax4
    ax4.imshow(signature)
    ax4.set_title('Signature')

    cv2.imwrite("./images/signature.jpg", signature)

    # Display the plot
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
