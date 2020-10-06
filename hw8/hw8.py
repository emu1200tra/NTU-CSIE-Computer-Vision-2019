import cv2
import random
import numpy as np

def readImg(filename='lena.bmp'):
    #read img
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return image

def snr(origImg, TargImg):
    origImg = origImg / 255.0
    TargImg = TargImg / 255.0
    meanOrig = np.mean(origImg)
    VS = np.mean(np.power(origImg - meanOrig, 2))
    meanTarg = np.mean(TargImg - origImg)
    VN = np.mean(np.power(TargImg - origImg - meanTarg, 2))

    return 20.0*np.log10(np.sqrt(VS)/np.sqrt(VN))

def gaussianNoise(img, amp):
    result = img.copy()
    w,h = img.shape
    for i in range(w):
        for j in range(h):
            result[i,j] = img[i,j] + int(amp * random.gauss(0,1))
    result = np.clip(result, 0, 255)
    cv2.imwrite("gaussNoise_"+str(amp)+".jpg", result)
    return result


def saltNoise(img, prob):
    result = img.copy()
    w,h = img.shape
    for i in range(w):
        for j in range(h):
            noise = random.uniform(0,1)
            if noise < prob:
                result[i,j] = 0
            elif noise > (1-prob):
                result[i,j] = 255
            else:
                result[i,j] = img[i,j]
    cv2.imwrite("saltnpepperNoise_"+str(prob).split('.')[1]+".jpg", result)
    return result    

def boxFilter(img, kernel, index):
    result = img.copy()
    w,h = result.shape
    border = int(kernel/2)
    img = np.pad(img, ((border,border),(border,border)), 'constant', constant_values=0)
    for i in range(w):
        for j in range(h):
            result[i,j] = int(np.sum(img[i:i+1+2*border,j:j+1+2*border])/kernel**2)
    cv2.imwrite("boxF_"+str(kernel)+"_"+str(index)+".jpg", result)
    return result

def medianFilter(img, kernel, index):
    result = img.copy()
    w,h = result.shape
    border = int(kernel/2)
    img = np.pad(img, ((border,border),(border,border)), 'constant', constant_values=0)
    for i in range(w):
        for j in range(h):
            tmp = img[i:i+1+2*border,j:j+1+2*border].flatten()
            tmp = np.sort(tmp).tolist()
            place = int(len(tmp)/2)
            result[i,j] = tmp[place]
    cv2.imwrite("medianF_"+str(kernel)+"_"+str(index)+".jpg", result)
    return result


def ImgPreProcess():
    #kernel
    Dilkernel = np.array([[0,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0]])
    Erokernel = np.array([[255,1,1,1,255],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[255,1,1,1,255]])

    return Dilkernel, Erokernel


def dilation(image, kernel, shape):
    #dilation on gray image
    dilImg = np.zeros(shape)
    w,h = image.shape
    for i in range(2,w-2):
        for j in range(2,h-2):
            dilImg[i-2, j-2] = np.amax(image[i-2:i+3, j-2:j+3]*kernel)
    return dilImg

def erosion(image, kernel, shape):
    #erosion on gray image
    eroImg = np.zeros(shape)
    w,h = image.shape
    for i in range(2,w-2):
        for j in range(2,h-2):
            eroImg[i-2, j-2] = np.amin(image[i-2:i+3, j-2:j+3]*kernel)
    return eroImg

def opening(image, Dilkernel, Erokernel, shape):
    #erosion than dilation
    image = np.pad(image, ((2,2),(2,2)), 'constant', constant_values=255)
    eroImg = erosion(image, Erokernel, shape)
    eroImg = np.pad(eroImg, ((2,2),(2,2)), 'constant', constant_values=0)
    openImg = dilation(eroImg, Dilkernel, shape)
    return openImg

def closing(image, Dilkernel, Erokernel, shape):
    #dilation than erosion
    image = np.pad(image, ((2,2),(2,2)), 'constant', constant_values=0)
    dilImg = dilation(image, Dilkernel, shape)
    dilImg = np.pad(dilImg, ((2,2),(2,2)), 'constant', constant_values=255)
    closImg = erosion(dilImg, Erokernel, shape)
    return closImg

def openThenClose(img, Dilkernel, Erokernel, shape, index):
    openImg = opening(img, Dilkernel, Erokernel, shape)
    closImg = closing(openImg, Dilkernel, Erokernel, shape)
    cv2.imwrite("openThenClose_"+str(index)+".jpg", closImg)
    return closImg

def closeThenOpen(img, Dilkernel, Erokernel, shape, index):
    closImg = closing(img, Dilkernel, Erokernel, shape)
    openImg = opening(closImg, Dilkernel, Erokernel, shape)
    cv2.imwrite("closeThenOpen_"+str(index)+".jpg", openImg)
    return openImg    

def main():
    img = readImg()
    shape = img.shape
    amp = [10, 30]
    gauss = []
    for i in amp:
        gauss.append(gaussianNoise(img.copy(), i))
        print("gauss {} snr: {}".format(i, snr(img, gauss[-1])))
    
    prob = [0.1, 0.05]
    salt = []
    for i in prob:
        salt.append(saltNoise(img.copy(), i))
        print("saltPepper {} snr: {}".format(i, snr(img, salt[-1])))
    kernel = [3,5]
    
    box = []
    for i in kernel:
        for j,k in zip(gauss, amp):
            box.append(boxFilter(j.copy(), i, k))
            print("box {} gauss {} snr: {}".format(i, k, snr(img, box[-1])))
    for i in kernel:
        for j,k in zip(salt, prob):
            box.append(boxFilter(j.copy(), i, k))
            print("box {} salt {} snr: {}".format(i, k, snr(img, box[-1])))

    median = []
    for i in kernel:
        for j,k in zip(gauss, amp):
            median.append(medianFilter(j.copy(), i, k))
            print("median {} gauss {} snr: {}".format(i, k, snr(img, median[-1])))            
    for i in kernel:
        for j,k in zip(salt, prob):
            median.append(medianFilter(j.copy(), i, k))
            print("median {} salt {} snr: {}".format(i, k, snr(img, median[-1])))            
    
    openThenCloseImgs = []
    closeThenOpenImgs = []
    Dilkernel, Erokernel = ImgPreProcess()
    for j,k in zip(gauss, amp):
        openThenCloseImgs.append(openThenClose(j.copy(), Dilkernel, Erokernel, shape, k))
        print("openThenClose gauss {} snr: {}".format(k, snr(img, openThenCloseImgs[-1])))                
        closeThenOpenImgs.append(closeThenOpen(j.copy(), Dilkernel, Erokernel, shape, k))
        print("closeThenOpen gauss {} snr: {}".format(k, snr(img, closeThenOpenImgs[-1])))

    for j,k in zip(salt, prob):
        openThenCloseImgs.append(openThenClose(j.copy(), Dilkernel, Erokernel, shape, k))
        print("openThenClose salt {} snr: {}".format(k, snr(img, openThenCloseImgs[-1]))) 
        closeThenOpenImgs.append(closeThenOpen(j.copy(), Dilkernel, Erokernel, shape, k))
        print("closeThenOpen salt {} snr: {}".format(k, snr(img, closeThenOpenImgs[-1])))


if __name__ == "__main__":
    main()