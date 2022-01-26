import os
import numpy as np
import cv2
import yaml
import albumentations as A
import random
import os_module as om
import re

script_path = os.path.dirname(__file__)
os.chdir(script_path)

def extract_frame(config):

    config_nm = config['extract_frame']

    cap = cv2.VideoCapture(config_nm['video_path'])
    first_name = os.path.splitext(os.path.basename(config_nm['video_path']))[0]

    frame_count = 0

    if not os.path.isdir(os.path.join(config_nm['out_path'])):
        os.mkdir(os.path.join(config_nm['out_path']))

    if not os.path.isdir(os.path.join(config_nm['out_path'], first_name)):
        os.mkdir(os.path.join(config_nm['out_path'], first_name))

    while(1):
        ret, frame = cap.read() 

        if not ret:
            break
        
        # X : 240 ~ 1680
        if frame_count % config_nm['save_frame_count'] == 0:

            # clean_img = frame[:,240:1680,:]
            clean_img = frame

            # frame_re = cv2.resize(clean_img, (config_nm['out_img_size'], config_nm['out_img_size']))
            cv2.imwrite(os.path.join(config_nm['out_path'], first_name, str(frame_count).zfill(5) + ".jpg"), clean_img)

            print("frame : {}".format(frame_count))

        frame_count += 1

def normal_making_ver2(config):

    config_nm = config['normal_making_ver2']

    if not os.path.isdir(config_nm['output_path']):
        os.mkdir(config_nm['output_path'])

    folders = om.extract_folder(config_nm['input_path'], full_path=True)

    for folder in folders:

        imgs_path = om.extract_folder(folder, ext=".jpg", full_path=True)

        for img_path in imgs_path:
            img = cv2.imread(img_path)

            h_, w_, c_ = img.shape
            # w_h_ratio = round(w_/h_,2)
            w_h_ratio = random.randrange(80,121) / 100

            img_resized = cv2.resize(img, (config_nm['origin_size'], config_nm['origin_size']))
            inher_number = os.path.splitext(os.path.split(img_path)[1])[0]

            if "hallway" in folder:
                for idx in range(config_nm['hallway_num']):
                    transform = A.Compose([A.RandomSizedCrop(min_max_height=[min(h_/2, 600),int(h_*0.8)], 
                        height=config_nm['crop_size'], 
                        width=config_nm['crop_size'], 
                        w2h_ratio=w_h_ratio), 
                        A.HorizontalFlip(p=0.4),
                        A.VerticalFlip(p=0.4),
                        A.Rotate(limit=[40,40], border_mode=1, p=0.3),
                        A.RGBShift(r_shift_limit = [-10,-10], g_shift_limit = [-20,-20], b_shift_limit = [10,10], p=0.2),
                        A.Downscale(scale_min=0.7, scale_max=0.7, p=0.2),
                        A.Blur(blur_limit=[4,4], p=0.3)])
                    transformed = transform(image=img)
                    f_name_ = os.path.join(config_nm['output_path'], "1_" + os.path.basename(folder) + "_" + inher_number + "_" + str(idx).zfill(2))
                    if not os.path.isdir(f_name_):
                        os.mkdir(f_name_)
                    
                    cv2.imwrite(os.path.join(f_name_, "crop.jpg"), transformed["image"])
                    cv2.imwrite(os.path.join(f_name_, "origin.jpg"), img_resized)
            else:
                #! 보통은 여기서 만들어짐 -- 복도제외
                for idx in range(config_nm['transform_num']):
                    transform = A.Compose([A.RandomSizedCrop(min_max_height=[min(h_/2, 600),int(h_*0.8)], 
                        height=config_nm['crop_size'], 
                        width=config_nm['crop_size'], 
                        w2h_ratio=w_h_ratio), 
                        A.HorizontalFlip(p=0.4),
                        A.VerticalFlip(p=0.4),
                        A.Rotate(limit=[40,40], border_mode=1, p=0.3),
                        A.RGBShift(r_shift_limit = [-10,-10], g_shift_limit = [-20,-20], b_shift_limit = [10,10], p=0.2),
                        A.Downscale(scale_min=0.7, scale_max=0.7, p=0.2),
                        A.Blur(blur_limit=[4,4], p=0.3)])
                    transformed = transform(image=img)
                    f_name_ = os.path.join(config_nm['output_path'], "1_" + os.path.basename(folder) + "_" + inher_number + "_" + str(idx).zfill(2))
                    if not os.path.isdir(f_name_):
                        os.mkdir(f_name_)
                    
                    cv2.imwrite(os.path.join(f_name_, "crop.jpg"), transformed["image"])
                    cv2.imwrite(os.path.join(f_name_, "origin.jpg"), img_resized)

            print("Img_ Generating ~~ : {}".format(img_path))

        print("Processing Complete {} folder ~~".format(folder))
        

def anormal_making_ver2(config):

    config_nm = config['anormal_making_ver2']

    if not os.path.isdir(config_nm['output_path']):
        os.mkdir(config_nm['output_path'])

    folders = om.extract_folder(config_nm['input_path'], full_path=True)

    for idx in range(config_nm['making_number']):

        while 1:
            f1, f2 = random.sample(folders, 2)
            dot_f1, dot_f2 = re.split("_", f1)[-1], re.split("_", f2)[-1]
            if dot_f1 != dot_f2:
                break

        origin_path = random.sample( om.extract_folder(f1, ext=".jpg", full_path=True), 1 )[0]
        cropped_path = random.sample( om.extract_folder(f2, ext=".jpg", full_path=True), 1 )[0]

        origin_img = cv2.imread(origin_path)
        cropped_img = cv2.imread(cropped_path)

        h_, w_, c_ = cropped_img.shape
        # w_h_ratio = round(w_/h_,2)
        w_h_ratio = random.randrange(80,121) / 100

        img_resized = cv2.resize(origin_img, (config_nm['origin_size'], config_nm['origin_size']))

        transform = A.Compose([A.RandomSizedCrop(min_max_height=[min(h_/2, 600),int(h_*0.8)], 
                        height=config_nm['crop_size'], 
                        width=config_nm['crop_size'], 
                        w2h_ratio=w_h_ratio), 
                        A.HorizontalFlip(p=0.4),
                        A.VerticalFlip(p=0.4),
                        A.Rotate(limit=[40,40], border_mode=1, p=0.3),
                        A.RGBShift(r_shift_limit = [-10,-10], g_shift_limit = [-20,-20], b_shift_limit = [10,10], p=0.2),
                        A.Downscale(scale_min=0.7, scale_max=0.7, p=0.2),
                        A.Blur(blur_limit=[4,4], p=0.3)])
        transformed = transform(image=cropped_img)
        f_name_ = os.path.join(config_nm['output_path'], "0_" + str(idx).zfill(5) + "_" + dot_f1 + "_" + dot_f2)
        if not os.path.isdir(f_name_):
            os.mkdir(f_name_)

        cv2.imwrite(os.path.join(f_name_, "crop.jpg"), transformed["image"])
        cv2.imwrite(os.path.join(f_name_, "origin.jpg"), img_resized)
        
        print("Process : {} | {}".format(idx, config_nm['making_number']))

if __name__ == "__main__":
    with open('data_maker.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # extract_frame(config)
    # normal_making_ver2(config)
    anormal_making_ver2(config)


