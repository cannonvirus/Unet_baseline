import cv2
import os
import sys
import re

script_path = os.path.dirname(__file__)
os.chdir(script_path)

import os_module as om


def main():
    
    fps = 20
    # width, height = 1920, 1080

    width, height = 640, 480
    fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out = cv2.VideoWriter('video_th03.avi', fcc, fps, (width, height))

    om.zfill_filename(path="/works/Unet_test/project_output/output_img_th03", zfill_num=4, ext=".jpg")
    result = om.extract_folder(path="/works/Unet_test/project_output/output_img_th03", ext=".jpg", full_path=True)
    
    for idx, path in enumerate(result):
        img = cv2.imread(path)
        img = cv2.resize(img, (640,480), interpolation=cv2.INTER_LINEAR)
        out.write(img)
        
    out.release()
    
    return 0


if __name__ == "__main__":
    main()


