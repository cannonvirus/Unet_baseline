import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml
import os
import imgsim
import time

script_path = os.path.dirname(__file__)
os.chdir(script_path)

def extract_video_features(video_cap, vtr, resizing=128, frame_interval=15):

    #! info
    frame_count = 0
    video_features = []

    while(1):
        ret, frame = video_cap.read() 

        if not ret:
            break
        
        if frame_count % frame_interval == 0:

            if frame_count == 0:
                h, w, c = frame.shape
                # po_w_0, po_w_1, po_w_2, po_w_3, po_w_4 = 0, int(w/4), int(w/2), int(3*w/4), w
                # po_h_0, po_h_1, po_h_2, po_h_3, po_h_4 = 0, int(h/4), int(h/2), int(3*h/4), h

                po_w_0, po_w_1, po_w_2, po_w_3, po_w_4, po_w_5, po_w_6, po_w_7, po_w_8 = [ int(i*w/8) for i in range(9)]
                po_h_0, po_h_1, po_h_2, po_h_3, po_h_4, po_h_5, po_h_6, po_h_7, po_h_8 = [ int(i*h/8) for i in range(9)]

                # area = np.array([[po_w_1, po_h_1, po_w_2, po_h_3],
                #                 [po_w_2, po_h_1, po_w_3, po_h_3],
                #                 [po_w_1, po_h_0, po_w_3, po_h_1],
                #                 [po_w_1, po_h_1, po_w_3, po_h_2],
                #                 [po_w_1, po_h_1, po_w_3, po_h_3]])

                # area = np.array([[po_w_2, po_h_2, po_w_4, po_h_6],
                #                 [po_w_4, po_h_2, po_w_6, po_h_6],
                #                 [po_w_2, po_h_0, po_w_6, po_h_2],
                #                 [po_w_2, po_h_2, po_w_6, po_h_4],
                #                 [po_w_2, po_h_2, po_w_6, po_h_6], 
                #                 [po_w_1, po_h_2, po_w_3, po_h_3]])

                # area = np.array([[po_w_2, po_h_2, po_w_4, po_h_6],
                #                 [po_w_4, po_h_2, po_w_6, po_h_6],
                #                 [po_w_2, po_h_0, po_w_6, po_h_2],
                #                 [po_w_2, po_h_2, po_w_6, po_h_6], 
                #                 [po_w_1, po_h_2, po_w_3, po_h_3]])
                
                area = np.array([[po_w_2, po_h_2, po_w_4, po_h_6],
                                [po_w_4, po_h_2, po_w_6, po_h_6],
                                [po_w_2, po_h_0, po_w_6, po_h_2],
                                [po_w_2, po_h_2, po_w_6, po_h_6], 
                                [po_w_1, po_h_2, po_w_3, po_h_3]])

            snap_list = []
            for ar in area:
                snap = frame[ar[1]:ar[3], ar[0]:ar[2], :]
                snap = cv2.resize(snap, (resizing, resizing))
                snap_list.append(snap)

            vec_features = vtr.vectorize(np.array(snap_list)) # grid X feature
            video_features.append(vec_features)

        frame_count += 1    

    #! video 0 frame 부터 시작하도록 초기화 작업
    video_cap.release()
    return np.array(video_features)

def extract_image_features(imgs_path, vtr, resizing=256):
    img_list = []
    for img_path in imgs_path:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (resizing, resizing))
        img_list.append(img)
    img_features = vtr.vectorize(np.array(img_list)) # set X feature
    
    return img_features

def extract_optimal_frameset(video_feature, image_feature, frame_interval=60 ):

    video_feature_sfgf = np.expand_dims(video_feature,axis=0).repeat(image_feature.shape[0], axis=0)
    image_feature_sfgf = np.expand_dims(image_feature, axis=(1,2)).repeat(video_feature.shape[0], axis=1).repeat(video_feature.shape[1], axis=2)

    score_box = np.sqrt(np.min(np.sum((video_feature_sfgf  - image_feature_sfgf) ** 2, axis=3), axis=2))
    result_index = np.argmin(score_box, axis=1) * frame_interval
    result_score = np.min(score_box, axis=1)

    return result_index, result_score


if __name__ == "__main__":
    with open('color_histogram.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    start_ = time.time()
    resque_img_path = ["/works/Anormal_Unet/AIGC_data_utils/set01_rescue_image01.jpg",
                        "/works/Anormal_Unet/AIGC_data_utils/set01_rescue_image02.jpg",
                        "/works/Anormal_Unet/AIGC_data_utils/set01_rescue_image03.jpg",
                        "/works/Anormal_Unet/AIGC_data_utils/set01_rescue_image04.jpg",
                        "/works/Anormal_Unet/AIGC_data_utils/set01_rescue_image05.jpg",
                        "/works/Anormal_Unet/AIGC_data_utils/set01_rescue_image06.jpg",
                        "/works/Anormal_Unet/AIGC_data_utils/set01_rescue_image07.jpg",
                        "/works/Anormal_Unet/AIGC_data_utils/set01_rescue_image08.jpg",
                        "/works/Anormal_Unet/AIGC_data_utils/set01_rescue_image09.jpg",
                        "/works/Anormal_Unet/AIGC_data_utils/set01_rescue_image10.jpg",
                        "/works/Anormal_Unet/AIGC_data_utils/set01_rescue_image11.jpg",
                        "/works/Anormal_Unet/AIGC_data_utils/set01_rescue_image12.jpg"]

    video_cap = cv2.VideoCapture(config['video_path'])
    vtr = imgsim.Vectorizer(device="cuda")
    resizing = 256


    set01_drone = {}
    set01_drone['set01_drone01'] = [660,1020,1500,1980,3900,4320]
    set01_drone['set01_drone02'] = [420,720,1200,1740,2820,3180]
    set01_drone['set01_drone03'] = [420,1560,2040,2580,4020,4620]

    dictionary_key = os.path.splitext(os.path.basename(config['video_path']))[0]
    time_list = set01_drone[dictionary_key]
    cut_threshold = 18.5

    video_feature = extract_video_features(video_cap, vtr, resizing=256, frame_interval=config['frame_interval'])
    image_feature = extract_image_features(resque_img_path, vtr, resizing=256)
    
    result = extract_optimal_frameset(video_feature = video_feature, image_feature = image_feature, frame_interval=config['frame_interval'])

    for idx, (num, score) in enumerate(zip(result[0], result[1])):
        if score > cut_threshold:
            print("img : {} | frame : {} | score : {:.2f} | result : NONE".format(resque_img_path[idx], num, score))
        else:
            if num in range(time_list[0], time_list[1]):
                print("img : {} | frame : {} | score : {:.2f} | result : KM".format(resque_img_path[idx], num, score))
            elif num in range(time_list[2], time_list[3]):
                print("img : {} | frame : {} | score : {:.2f} | result : SS".format(resque_img_path[idx], num, score))
            elif num in range(time_list[4], time_list[5]):
                print("img : {} | frame : {} | score : {:.2f} | result : JL".format(resque_img_path[idx], num, score))
            else:
                print("img : {} | frame : {} | score : {:.2f} | result : NONE".format(resque_img_path[idx], num, score))

    print("time : {}".format(time.time() - start_ ))



