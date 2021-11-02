import os
import os_module as om
import random
import shutil

out_path = "/works/Anormal_Unet/test_pretrain"

for path in om.extract_folder("/works/Anormal_Unet/AIGC_pretrain_data", full_path=True):
    prob = random.randint(0,100)
    if prob <= 3:
        src = path
        dir_ = os.path.join(out_path, os.path.basename(path))
        # print(src, dir_)
        shutil.move(src, dir_)