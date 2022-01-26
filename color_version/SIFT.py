#%%
import numpy as np
import cv2
from matplotlib import pyplot as plt
img1 = cv2.imread('/works/Anormal_Unet/AIGC_data_utils/set01_rescue_image03.jpg',0)
img2 = cv2.imread('/works/Anormal_Unet/AIGC_data_utils/set01_drone01_60f/00960.jpg',0)[300:800,1000:1700]

method = 'ORB'  # 'SIFT'
lowe_ratio = 0.89

if method   == 'ORB':
    finder = cv2.ORB_create()
elif method == 'SIFT':
    finder = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = finder.detectAndCompute(img1,None)
kp2, des2 = finder.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []

for m,n in matches:
    if m.distance < lowe_ratio*n.distance:
        good.append([m])

msg1 = 'using %s with lowe_ratio %.2f' % (method, lowe_ratio)
msg2 = 'there are %d good matches' % (len(good))

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good, None, flags=2)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img3,msg1,(10, 250), font, 0.5,(255,255,255),1,cv2.LINE_AA)
cv2.putText(img3,msg2,(10, 270), font, 0.5,(255,255,255),1,cv2.LINE_AA)

plt.imshow(img3),plt.show()
# %%
