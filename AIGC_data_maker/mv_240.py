import os
import os_module as om
import re
import cv2
import shutil

def main():
    main_path = "/works/Anormal_Unet/AIGC_data_utils"
    f240_path = "/works/Anormal_Unet/AIGC_data_utils/classification_15f"
    f1080_path = "/works/Anormal_Unet/AIGC_data_utils/classification_15f_high"
    for folder in om.extract_folder(f240_path, full_path=True):

        target_folder = os.path.join(f1080_path, os.path.basename(folder))
        if not os.path.isdir( target_folder ):
            os.mkdir( target_folder )

        ta = os.path.basename(target_folder)
        origin_folder_name = re.split("_", ta)[0] + "_" + re.split("_", ta)[1]
        origin_path = os.path.join(main_path, origin_folder_name)
        
        for img_path in om.extract_folder(folder, ext=".jpg", full_path=True):
            img_name = os.path.basename(img_path)
            src = os.path.join(origin_path, img_name)
            drc = os.path.join(f1080_path, ta, img_name)
            print(src, drc)
            shutil.copy(src, drc)

if __name__ == "__main__":
    main()