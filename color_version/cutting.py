#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml
import os
import imgsim
import time


# img_path = "/works/Anormal_Unet/AIGC_data_utils/set01_drone01/01720.jpg"
img_path = "/works/Anormal_Unet/AIGC_data_utils/set01_drone03/02125.jpg"
resque_path = "/works/Anormal_Unet/AIGC_data_utils/set01_rescue_image11.jpg"
img = cv2.imread(img_path)
resque_img = cv2.imread(resque_path)

resizing = 256

resque_img_resized = cv2.resize(resque_img, (resizing, resizing))
print("resque")
plt.imshow(resque_img_resized)
plt.show()



h, w, c = img.shape

# po_w_0, po_w_1, po_w_2, po_w_3, po_w_4 = 0, int(w/4), int(w/2), int(3*w/4), w
# po_h_0, po_h_1, po_h_2, po_h_3, po_h_4 = 0, int(h/4), int(h/2), int(3*h/4), h
po_w_0, po_w_1, po_w_2, po_w_3, po_w_4, po_w_5, po_w_6, po_w_7, po_w_8 = [ int(i*w/8) for i in range(9)]
po_h_0, po_h_1, po_h_2, po_h_3, po_h_4, po_h_5, po_h_6, po_h_7, po_h_8 = [ int(i*h/8) for i in range(9)]

# area = np.array([[po_w_1, po_h_1, po_w_2, po_h_3],
#                 [po_w_2, po_h_1, po_w_3, po_h_3],
#                 [po_w_1, po_h_0, po_w_3, po_h_1],
#                 [po_w_1, po_h_1, po_w_3, po_h_2],
#                 [po_w_1, po_h_1, po_w_3, po_h_3], 
#                 [po_w_1, po_h_2, po_w_3, po_h_3]])

area = np.array([[po_w_2, po_h_2, po_w_4, po_h_6],
                [po_w_4, po_h_2, po_w_6, po_h_6],
                [po_w_2, po_h_0, po_w_6, po_h_2],
                [po_w_2, po_h_2, po_w_6, po_h_6], 
                [po_w_1, po_h_2, po_w_3, po_h_3],
                [po_w_0, po_h_1, po_w_2, po_h_6]])

vtr = imgsim.Vectorizer(device="cuda")
vec0 = vtr.vectorize(resque_img_resized)

for idx, ar in enumerate(area):
    snap = img[ar[1]:ar[3], ar[0]:ar[2], :]
    snap = cv2.resize(snap, (resizing, resizing))
    vec1 = vtr.vectorize(snap)
    dist = imgsim.distance(vec0, vec1)
    print("dist : {:.2f} | idx : {} | ar : {}".format(dist, idx, ar))
    plt.imshow(snap)
    plt.show()


# %%
