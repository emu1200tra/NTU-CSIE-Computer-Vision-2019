import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

def readImg(filename='lena.bmp'):
    #read img
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    print('shape:', image.shape)
    #binarize
    index = np.where(image >= 128)
    binary = np.zeros(image.shape)
    binary[index] = 255
    cv2.imwrite('binary.jpg', binary)
    return image, binary

def downSample(image):
    #down sample to 64*64
    w,h = image.shape
    result = np.zeros((int(w/8), int(h/8)))
    for i in range(0,w,8):
        for j in range(0,h,8):
            result[int(i/8),int(j/8)] = image[i,j]
    cv2.imwrite('downSample.jpg', result)
    return result

def hFunction(b, c, d, e):
    if b == c and (d != b or e != b):
        return "q"
    elif b == c and (d == b and e == b):
        return "r"
    else:
        return "s"

def counter(record):
    countQ = 0
    countR = 0
    for i in record:
        if i == "q":
            countQ += 1
        elif i == "r":
            countR += 1
    if countR == 4:
        return 5
    else:
        return countQ

def yokoi(image):
    image = np.pad(image, ((1,1),(1,1)), 'constant', constant_values=0)
    print('padding shape:', image.shape)
    w, h = image.shape
    with open('output.txt', "w") as f:
        for i in range(1, w-1):
            for j in range(1, h-1):
                if image[i,j] == 255:
                    record = []
                    record.append(hFunction(image[i,j], image[i+1,j], image[i+1,j-1], image[i,j-1]))
                    record.append(hFunction(image[i,j], image[i,j-1], image[i-1,j-1], image[i-1,j]))
                    record.append(hFunction(image[i,j], image[i-1,j], image[i-1,j+1], image[i,j+1]))
                    record.append(hFunction(image[i,j], image[i,j+1], image[i+1,j+1], image[i+1,j]))
                    num = counter(record)
                    f.write(str(num))
                else:
                    f.write(' ')
            f.write('\n')


if __name__ == "__main__":
    image, binary = readImg()
    down = downSample(binary)
    yokoi(down)

    
