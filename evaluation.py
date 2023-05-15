from skimage.metrics import structural_similarity as ssim
import cv2

def PSNRandSSIM(img1,img2,name):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim_score = format( ssim(gray1, gray2, full=True)[0]*100, '.2f')
    print(name)
    print("SSIM: {}".format(ssim_score))

    psnr =format( cv2.PSNR(img1, img2), '.2f')
    print("PSNR: {}".format(psnr))


LBS_input = cv2.imread("input/LSBimage.jpg")
LBS_output = cv2.imread("output/LSBimage_output.jpg")
DCT_input = cv2.imread("input/DCTfood.jpg")
DCT_output = cv2.imread("output/DCTwatermarked.jpg")
DFT_input = cv2.imread("input/DFTimage.jpg")
DFT_output = cv2.imread("output/DFTwatermarked.jpg")
DWT_input = cv2.imread("input/DWTcover.jpg")
DWT_output = cv2.imread("output/DWTwatermarked.jpg")
SVD_input = cv2.imread("input/SVDimage.jpg")
SVD_output = cv2.imread("output/SVDimage_output.jpg")


PSNRandSSIM(LBS_input,LBS_output, "LSB")
PSNRandSSIM(DCT_input,DCT_output, "DCT")
PSNRandSSIM(DFT_input,DFT_output, "DFT")
PSNRandSSIM(DWT_input,DWT_output, "DWT")
PSNRandSSIM(SVD_input,SVD_output, "SVD")

