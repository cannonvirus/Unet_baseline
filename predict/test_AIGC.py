import argparse
import logging
import os
import sys
from os import listdir
import yaml

sys.path.append('/works/Anormal_Unet/')

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet, UNet_ENC, UNet_ENC_Double, UNet_ENC_Double_Up, UNet_ENTRY_ENS, UNet_AIGC_ver2
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
import cv2



def predict_img(net,
                video_path,
                snap_shot_path,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    cap = cv2.VideoCapture(config['video_path'])
    frame_count = 0

    crop_img = cv2.imread(snap_shot_path)
    cv2.imwrite(os.path.join("/works/Anormal_Unet/trash", "crop.jpg"), crop_img)

    while(1):
        ret, frame = cap.read() 

        if not ret:
            break
        
        # X : 240 ~ 1680
        if frame_count % config['save_frame_count'] == 0:

            clean_img = frame[:,240:1680,:]
            img_stack = BasicDataset.AIGC_process(clean_img, crop_img, scale_factor)

            img_stack = img_stack.unsqueeze(0)
            img_stack = img_stack.to(device=device, dtype=torch.float32)
            
            ab_status = net(img_stack)
            prob = ab_status.cpu().detach().numpy()[0][0]
            
            
            if prob > 0.5:
                print("Prob : {:.2f} | frame : {}".format(prob, frame_count))
                cv2.imwrite(os.path.join("/works/Anormal_Unet/trash", "frame_{}.jpg").format(frame_count), clean_img)

        frame_count += 1
    
    return ab_status

def mask_to_image(mask, bgr2rgb):
    image = (mask.cpu().detach().numpy()*255).astype("uint8").squeeze(0).transpose(1,2,0)
    if bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_yaml(yaml_path):
    with open(yaml_path) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    return config_dict

if __name__ == "__main__":
    config = get_yaml("./predict/test_AIGC.yaml")
    # net = UNet_ENC(n_channels=6, n_classes=3, half_model=False)
    net = UNet_ENC_Double_Up(n_channels=3, n_classes=3, half_model=False)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:7')
    net.to(device=device)
    net.load_state_dict(torch.load(config['model_path'], map_location=device))

    mask = predict_img(net=net,
                        video_path=config['video_path'],
                        snap_shot_path=config['snap_shot_path'],
                        scale_factor=config['scale'],
                        out_threshold=config['threshold'],
                        device=device)
