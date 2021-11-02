import argparse
import logging
import os
import yaml
import cv2
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet, UNet_ENC, UNet_ENC_Double, UNet_ENC_Double_Up, UNet_ENTRY_ENS, UNet_AIGC_ver2

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from utils.Focal_Loss import WeightedFocalLoss, FocalLoss

import torch.nn.functional as F
import horovod.torch as hvd

def get_yaml():
    with open('config.yaml') as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    return config_dict

def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    
    # 1. Create dataset
    args = get_yaml()
    save_cp = args['save_checkpoints']
    
    dataset = BasicDataset(args['input_img_path'],
                           args['scale'],
                           args['input_time_series'])
    
    n_val = int(len(dataset) * args['validation_ratio'])
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    
    # ANCHOR : horovod
    if args['horovod']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train,
            num_replicas=hvd.size(),
            rank=hvd.rank()
            )

        train_loader_args = dict(batch_size=args['batch_size'], num_workers=args['num_worker'], \
            pin_memory=False, sampler=train_sampler)

        train_loader = DataLoader(train, **train_loader_args)
    else:
        train_loader_args = dict(batch_size=args['batch_size'], num_workers=args['num_worker'], pin_memory=False)
        train_loader = DataLoader(train, shuffle=True, **train_loader_args)


    val_loader_args = dict(batch_size=args['batch_size'], num_workers=args['num_worker'], pin_memory=False)  
    val_loader = DataLoader(val, drop_last=True, **val_loader_args)
    
    writer = SummaryWriter(comment=f'LR_{args["learning_rate"]}_BS_{batch_size}_SCALE_{args["scale"]}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {args["epoch"]}
        Batch size:      {batch_size}
        Learning rate:   {args["learning_rate"]}
        Training size:   {len(train)}
        Validation size: {len(val)}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {args["scale"]}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMPoptimizer.zero_grad()
    # optimizer = optim.RMSprop(net.parameters(), lr=args["learning_rate"], weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=args["learning_rate"], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-8, amsgrad=False)
    # ANCHOR : horovod
    if args['horovod']:
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=net.named_parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)
    

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.MSELoss().to(device)
    criterion_BCE = nn.BCELoss().to(device)
    # criterion_Focal = WeightedFocalLoss(alpha=.75).to(device)
    criterion_Focal = FocalLoss().to(device)

    # ANCHOR : horovod
    if args['horovod']:
        hvd.broadcast_parameters(
            net.state_dict(),
            root_rank=0)

    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=len(train), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image'].to(device=device, dtype=torch.float32)
                crops = batch['crop'].to(device=device, dtype=torch.float32)
                true_labels = batch['label'].to(device=device, dtype=torch.float32)

                anormal_pred = net(imgs, crops)
                loss = criterion_BCE(anormal_pred, true_labels.reshape(-1,1))
                
                
                # if not args["freezing_mode"]:
                #     loss = criterion_BCE(anormal_pred, true_labels.reshape(-1,1))

                # else:
                #     if args['focal_loss']:
                #         loss = criterion_Focal(anormal_pred, true_labels.reshape(-1,1))
                #     else:
                #         loss = criterion_BCE(anormal_pred, true_labels.reshape(-1,1))

                # ANCHOR : horovod
                if args['horovod']:
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad(set_to_none=True)
                
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                # loss.backward()
                # optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                if global_step % int(len(train) * args['validation_ratio']) == 0 and hvd.rank() == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        # writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device, batch_size, freeze_mode = args["freezing_mode"], config = args)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    logging.info('Validation Dice Coeff: {:.4f}'.format(val_score))
                    writer.add_scalar('Dice/test', val_score, global_step)

        if args['horovod']:
            if save_cp and epoch % args['save_model_epoch'] == 0 and hvd.rank() == 0:
                try:
                    os.mkdir(args["checkpoint_path"])
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(net.state_dict(),
                        args["checkpoint_path"] + f'Entry_1101_{epoch}.pth')
                logging.info(f'Checkpoint {epoch} saved !')
        else:
            if save_cp and epoch % args['save_model_epoch'] == 0:
                try:
                    os.mkdir(args["checkpoint_path"])
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(net.state_dict(),
                        args["checkpoint_path"] + f'Cow_epoch{epoch}.pth')
                logging.info(f'Checkpoint {epoch} saved !')
            
    writer.close()


if __name__ == '__main__':
    args = get_yaml()

    #* horovod 초기화
    if args['horovod']:
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # net = UNet_ENC_Double(n_channels=3, n_classes=3, bilinear=True, half_model=args['half_mode'], scale = args['scale'])
    net = UNet_AIGC_ver2(n_channels=3, n_classes=3, bilinear=True, half_model=args['half_mode'], scale = args['scale'])
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args["load_pretrain"]:
        net.load_state_dict(torch.load(args["load_pretrain"], map_location=device))
        logging.info(f'Model loaded from {args["load_pretrain"]}')

    net.to(device=device)
    
    if args["freezing_mode"]:
        logging.info(f'Mode : Freezing mode ~!!')
        for idx, child in enumerate(net.children()):
            if child._get_name() in ("DoubleConv", "Down", "Up", "OutConv"):
                for param in child.parameters():
                    param.requires_grad = False
    
    try:
        train_net(net=net,
                  epochs=args["epoch"],
                  batch_size=args["batch_size"],
                  learning_rate=args["learning_rate"],
                  device=device,
                  img_scale=args["learning_rate"],
                  val_percent=args["validation_ratio"],
                  amp=args["amp"])
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)