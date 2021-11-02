from os.path import splitext
from os import listdir
import os
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import imageio
import random
import re
import cv2

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, scale=128, time_series=4, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.scale = scale
        self.time_series = time_series
        self.mask_suffix = mask_suffix
        
        self.ids = []
        for kd in listdir(imgs_dir):
            if len(listdir(os.path.join(imgs_dir, kd))) > self.time_series:
                self.ids.append(os.path.join(self.imgs_dir, kd))    
                
            # if re.search(".png", kd) is not None or re.search(".jpg", kd) is not None:
            #     self.ids.append(os.path.join(self.imgs_dir, kd))
                
        self.ids = sorted(self.ids)
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img, scale, bgr2rgb = True, float32 = True):
        # FIXME : 가변변수로 받을 수 있도록
        h,w,c = img.shape
        # if w > scale or h > scale:
        #     img = cv2.resize(img, dsize=(scale, scale), interpolation=cv2.INTER_AREA)

        # if len(img.shape) == 2:
        #     img_nd = np.expand_dims(img, axis=2)
            
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        img = torch.from_numpy(img.transpose(2, 0, 1))
        
        if float32:
            img = img.float()

        # HWC to CHW
        if img.max() > 1:
            img = img / 255

        return img
    
    @classmethod
    def time_series_preprocess(cls, img_list, scale, bgr2rgb = True, float32 = True):
        # FIXME : 가변변수로 받을 수 있도록
        img_stack = None
        
        for img_path in img_list:
            img = cv2.imread(img_path)
            h,w,c = img.shape
    
            if w != scale or h != scale:
                img = cv2.resize(img, dsize=(scale, scale), interpolation=cv2.INTER_AREA)

            if len(img.shape) == 2:
                img_nd = np.expand_dims(img, axis=2)
                
            if img.shape[2] == 3 and bgr2rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            if img_stack is None:
                img_stack = img.transpose(2, 0, 1)
            else:
                img = img.transpose(2, 0, 1)
                img_stack = np.vstack([img_stack, img])
            
        img = torch.from_numpy(img_stack)
        
        if float32:
            img = img.float()

        # HWC to CHW
        if img.max() > 1:
            img_final = img / 255

        return img_final

    @classmethod
    def two_img_preprocess(cls, background, crop_img, scale, bgr2rgb = True, float32 = True):
        # FIXME : 가변변수로 받을 수 있도록
        img_stack = None
        
        for img_path in [background, crop_img]:
            img = cv2.imread(img_path)
            h,w,c = img.shape
    
            if w != scale or h != scale:
                img = cv2.resize(img, dsize=(scale, scale), interpolation=cv2.INTER_AREA)

            if len(img.shape) == 2:
                img_nd = np.expand_dims(img, axis=2)
                
            if img.shape[2] == 3 and bgr2rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            if img_stack is None:
                img_stack = img.transpose(2, 0, 1)
            else:
                img = img.transpose(2, 0, 1)
                img_stack = np.vstack([img_stack, img])
            
        img = torch.from_numpy(img_stack)
        
        if float32:
            img = img.float()

        # HWC to CHW
        if img.max() > 1:
            img_final = img / 255

        return img_final

    @classmethod
    def two_img_preprocess_ENS(cls, background, crop_img, scale, bgr2rgb = True, float32 = True):
        # FIXME : 가변변수로 받을 수 있도록
        img_stack = None

        Edge_point_hw = [(0,0),(0,240),(0,480),(120,120),(120,240),(120,360),(240,0),(240,240),(240,480)]

        
        for img_path in [background, crop_img]:
            img = cv2.imread(img_path)
            h,w,c = img.shape
    
            if w != scale or h != scale: # ORIGIN
                # img = cv2.resize(img, dsize=(scale, scale), interpolation=cv2.INTER_AREA)
                for (yp, xp) in Edge_point_hw:

                    img_point = img[yp:yp+scale, xp:xp+scale, :]
                        
                    if img.shape[2] == 3 and bgr2rgb:
                        img_point = cv2.cvtColor(img_point, cv2.COLOR_BGR2RGB)
                        
                    if img_stack is None:
                        img_stack = img_point.transpose(2, 0, 1)
                    else:
                        img_point = img_point.transpose(2, 0, 1)
                        img_stack = np.vstack([img_stack, img_point])

            else:
                if len(img.shape) == 2:
                    img_nd = np.expand_dims(img, axis=2)
                    
                if img.shape[2] == 3 and bgr2rgb:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                if img_stack is None:
                    img_stack = img.transpose(2, 0, 1)
                else:
                    img = img.transpose(2, 0, 1)
                    img_stack = np.vstack([img_stack, img])

            
        img = torch.from_numpy(img_stack)
        
        if float32:
            img = img.float()

        if img.max() > 1:
            img_final = img / 255

        return img_final

    @classmethod
    def AIGC_process(cls, background, crop_img, scale, bgr2rgb = True, float32 = True):
        # FIXME : 가변변수로 받을 수 있도록
        img_stack = None

        #? background 
        h,w,c = background.shape

        if w != scale or h != scale:
            background = cv2.resize(background, dsize=(scale, scale), interpolation=cv2.INTER_AREA)

        if background.shape[2] == 3 and bgr2rgb:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

        img_stack = background.transpose(2, 0, 1)

        #? crop_img
        h,w,c = crop_img.shape

        if w != scale or h != scale:
            crop_img = cv2.resize(crop_img, dsize=(scale, scale), interpolation=cv2.INTER_AREA)

        if crop_img.shape[2] == 3 and bgr2rgb:
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

        crop_img = crop_img.transpose(2, 0, 1)
        img_stack = np.vstack([img_stack, crop_img])
            
        img = torch.from_numpy(img_stack)
        
        if float32:
            img = img.float()

        # HWC to CHW
        if img.max() > 1:
            img_final = img / 255

        return img_final

    def __getitem__(self, i):
        
        pig_series_path = self.ids[i]
        
        label = 0
        
        if int(re.split("_",re.split("/", pig_series_path)[-1])[0]) == 0:
            label = 0
        else:
            label = 1
        
        
        background_path = os.path.join(pig_series_path, "origin.jpg")
        crop_img_path = os.path.join(pig_series_path, "crop.jpg")

        # img_cc = self.two_img_preprocess(background_path, crop_img_path, self.scale)
        # img_cc = self.two_img_preprocess_ENS(background_path, crop_img_path, self.scale)

        background_img = cv2.imread(background_path)
        crop_img = cv2.imread(crop_img_path)

        img_cc = self.preprocess(img=background_img, scale=240)
        crp_cc = self.preprocess(img=crop_img, scale=120)

        return {
            'image': img_cc,
            'crop' : crp_cc,
            'label' : label
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, scale=1):
        super().__init__(imgs_dir, scale, mask_suffix='_mask')
