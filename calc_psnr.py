import cv2
img1 = cv2.imread('/home/featurize/work/RawFomer/static/enhanced_19700101_08_03_03_1920_1280_3480.bmp')
img2 = cv2.imread('/home/featurize/data/RGB/19700101_08_03_03_1920_1280_3480.bmp')
psnr = cv2.PSNR(img1, img2)
print(psnr)