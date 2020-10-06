import cv2
import random
import numpy as np


def snr(origImg, TargImg):
    origImg = origImg / 255.0
    TargImg = TargImg / 255.0
    meanOrig = np.mean(origImg)
    VS = np.mean(np.power(origImg - meanOrig, 2))
    meanTarg = np.mean(TargImg - origImg)
    VN = np.mean(np.power(TargImg - origImg - meanTarg, 2))
    return 20.0*np.log10(np.sqrt(VS)/np.sqrt(VN))


img1 = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("median_5x5.bmp", cv2.IMREAD_GRAYSCALE)
print(snr(img1, img2))