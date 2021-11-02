import torch
import torch.nn.functional as F
from tqdm import tqdm

import torch.nn as nn
import numpy as np
from utils.Focal_Loss import WeightedFocalLoss, FocalLoss

def eval_net(net, loader, device, batch_size, freeze_mode, config): 
    """Evaluation without the densecrf with the dice coefficient"""
    
    net.eval()
    # mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    criterion = nn.MSELoss().to(device)
    criterion_BCE = nn.BCELoss().to(device)
    criterion_Focal = FocalLoss().to(device)

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, crops, true_labels = batch['image'], batch['crop'], batch['label']
            # imgs, true_labels = batch['image'], batch['label']
            
            imgs = imgs.to(device=device, dtype=torch.float32)
            crops = crops.to(device=device, dtype=torch.float32)
            true_labels = true_labels.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                anormal_pred = net(imgs, crops)
                loss = criterion_BCE(anormal_pred, true_labels.reshape(-1,1))
                
                # if not freeze_mode:
                #     loss = criterion_BCE(anormal_pred, true_labels.reshape(-1,1))

                # else:
                #     if config['focal_loss']:
                #         loss = criterion_Focal(anormal_pred, true_labels.reshape(-1,1))
                #     else:
                #         loss = criterion_BCE(anormal_pred, true_labels.reshape(-1,1))


            tot += loss
            pbar.update()

    net.train()
    return tot / n_val
