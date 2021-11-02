import os
import sys
import numpy as np
import cv2
import yaml
import pandas as pd
import pandasql as ps
import math

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from unet import UNet

script_path = os.path.dirname(__file__)
os.chdir(script_path)


def get_yaml():
    with open('out_config.yaml') as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    return config_dict


def DB_intergrity_check(pd_det, frame_list):
    
    result = True
    
    for idx, frm in enumerate(frame_list):
        pd_frame_det = ps.sqldf("select DISTINCT frame, trk, x, y, w, h, t from pd_det where frame = {}".format(frm))
        
        #? DB 무결성 검사 코드 : 한 프레임에 trk는 한 행씩 존재해야함. 만약 result 값이 있는 경우에는 뭔가 잘못된거임
        result = ps.sqldf("select trk, COUNT(*) from pd_frame_det group by trk having COUNT(*) != 1".format(frm))
        if len(result.to_dict()['trk'].values()) > 0:
            result = False
            break
        
        print("process : {} | {}".format(idx, len(frame_list)))
        
    return result

def cutter_fix_45(xc, yc, width, height):
    f_dist = (width + height) / np.sqrt(2)
    answer_dict_ = {
        "xmin" : int(xc - f_dist/2),
        "ymin" : int(yc - f_dist/2),
        "xmax" : int(xc + f_dist/2),
        "ymax" : int(yc + f_dist/2)
    }
    return answer_dict_

def roi_in_box(img_width, img_height, rbox_dict, width_pad, height_pad):
    min_x = rbox_dict["xmin"]
    min_y = rbox_dict["ymin"]
    max_x = rbox_dict["xmax"]
    max_y = rbox_dict["ymax"]

    width_boundary = (img_width * width_pad, img_width * (1-width_pad))
    height_boundary = (img_height * height_pad, img_height * (1-height_pad))

    if min_x >= width_boundary[0] and max_x <= width_boundary[1] and \
        min_y >= height_boundary[0] and max_y <= height_boundary[1]:
        return True
    else:
        return False


def image_stack_process(stack_img, img, timeseries, bgr2rgb = True):
    
    if img.shape[2] == 3 and bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    img_stack = np.vstack([stack_img, img.transpose(2, 0, 1)])

    if img_stack.shape[0] > timeseries * 3:
        img_stack = img_stack[3:,:,:]
    
    
    return img_stack

def mask_to_image(mask, bgr2rgb=True):
    image = (mask.cpu().detach().numpy()*255).astype("uint8").squeeze(0).transpose(1,2,0)
    if bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def main():
    config = get_yaml()
    
    #* 모델 불러오기
    net = UNet(n_channels=12, n_classes=3, half_model=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(config['load_data']['model_path'], map_location=device))
    net.eval()

    
    det = pd.read_csv(config['load_data']['det_folder'], sep=",",encoding="utf-8")
    
    pd_database = ps.sqldf("select distinct frame, trk, x, y, w, h, t from det")
    
    pd_frame_list = ps.sqldf("select distinct frame from det")
    frame_list = np.fromiter(pd_frame_list.to_dict()['frame'].values(), dtype=int)
    
    if config['load_data']['integrity_check']:
        check_output = DB_intergrity_check(pd_database, frame_list)
        if not check_output:
            print("***** integrity violation *****")
            return False
        
    dict_info_ = {}

    print("======================= step 2 =======================")
    for idx, frm in enumerate(frame_list):
        print("frame process : {} | {}".format(idx, len(frame_list)))
        
        #* frame meta data
        pd_frame_det = ps.sqldf("select * from pd_database where frame = {}".format(frm))
        np_trk_list = np.fromiter( pd_frame_det["trk"].to_dict().values(), dtype=int )
        
        #* image loader
        img = cv2.imread(os.path.join(config['load_data']['img_folder'], "{}{}".format(frm, config['load_data']['img_format'])))
        img_h, img_w, _ = img.shape
        
        img_out = img.copy()
        
        for jdx, trk in enumerate(np_trk_list):
            # print("  tracker process : {} | {}".format(jdx, len(np_trk_list)))
            pd_frame_trk_det = ps.sqldf("select distinct x,y,w,h,t,frame,trk from pd_frame_det where trk = {}".format(trk))
            
            x_cen  = int(pd_frame_trk_det['x'])
            y_cen  = int(pd_frame_trk_det['y'])
            width  = int(pd_frame_trk_det['w'])
            height = int(pd_frame_trk_det['h'])
            theta  = float(pd_frame_trk_det['t'])
            
            Rfdot = cutter_fix_45(x_cen, y_cen, width, height)
            
            #* 주요 관심구역 안에 있는가?
            if roi_in_box(img_w, img_h, Rfdot, config['load_data']['width_roi_padding'], config['load_data']['height_roi_padding']):
                pig_img = img[Rfdot['ymin']:Rfdot['ymax'], Rfdot['xmin']:Rfdot['xmax'],:]
                pig_img = cv2.resize(pig_img, (config['load_data']['save_img_scale'],config['load_data']['save_img_scale']), interpolation=cv2.INTER_LINEAR)
                
                #* input image makeing
                if trk not in dict_info_.keys():
                    dict_info_[trk] = {}
                    dict_info_[trk]["input"] = pig_img.transpose(2, 0, 1)
                else:
                    imgs_value = image_stack_process(dict_info_[trk]["input"], pig_img, config['load_data']['time_series'])
                    dict_info_[trk]["input"] = imgs_value

                dict_info_[trk]["frame"] = frm
                dict_info_[trk]["answer"] = pig_img
                                
                #* timeseries INPUT값 조건 충족 하는지?
                if dict_info_[trk]["input"].shape[0] == config['load_data']['time_series'] * 3:
                    torch_img = torch.from_numpy(dict_info_[trk]["input"])
                    
                    if config['load_data']['float32']:
                        torch_img = torch_img.float()
                        
                    torch_img = torch_img / 255
                    torch_img = torch_img.unsqueeze(0)
                    torch_img = torch_img.to(device=device, dtype=torch.float32)
                    output, prod_anormal = net(torch_img)
                    
                    pred_img = mask_to_image(output)
                    
                    if config['load_data']['devide_mode'] == "psnr":
                    
                        MSE = np.mean((pig_img - pred_img) ** 2)
                        PSNR = 20 * math.log10(255.0 / np.sqrt(MSE))
                        
                        if PSNR > config['load_data']['psnr']:
                            img_out = cv2.ellipse(img_out, (x_cen, y_cen), (int(width/2), int(height/2)), (theta*180/3.14), 0, 360, (255,0,0), 3 )
                        else:
                            img_out = cv2.ellipse(img_out, (x_cen, y_cen), (int(width/2), int(height/2)), (theta*180/3.14), 0, 360, (0,0,255), 3 )
                    else:
                        anb_prob = prod_anormal.cpu().detach().numpy()[0][0]
                        pred_answer = 1 if anb_prob > config['load_data']['prob_threshold'] else 0
                        if pred_answer == 1:
                            img_out = cv2.ellipse(img_out, (x_cen, y_cen), (int(width/2), int(height/2)), (theta*180/3.14), 0, 360, (255,0,0), 3 )
                        else:
                            img_out = cv2.ellipse(img_out, (x_cen, y_cen), (int(width/2), int(height/2)), (theta*180/3.14), 0, 360, (0,0,255), 3 )

        if idx == 0:
            if not os.path.isdir(config['load_data']["output_img_path"]):
                os.mkdir(config['load_data']["output_img_path"])
        cv2.imwrite( os.path.join( config['load_data']["output_img_path"], "{}.jpg".format(frm) ), img_out)
    
if __name__ == "__main__":
    main()