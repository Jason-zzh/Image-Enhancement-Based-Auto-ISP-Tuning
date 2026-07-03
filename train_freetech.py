import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as PSNR
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import tqdm
import glob
import time
import datetime
from model_freetech import RawFormer
from load_dataset_freetch import load_data_RawRGB_Paired


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

if __name__ == '__main__':
    opt = {}
    opt={'base_lr':1e-4}        # base learning rate
    opt['gpu'] = '0'
    opt['batch_size'] = 16      # batch size
    opt['dataset'] = 'Freetech'      # SID/MCR dataset
    opt['patch_size'] = 512     # cropped image patch size when training
    opt['model_size'] = 'S'     # model size, small/base/large --> 32/48/64
    opt['epochs'] = 1000        # total training epochs

    # These are folders
    save_weights_file = os.path.join('result', opt['dataset'], 'weights')   # save trained models
    save_images_file = os.path.join('result', opt['dataset'], 'images')     # save tested images
    save_csv_file = os.path.join('result', opt['dataset'], 'csv')           # save tested images' psnr/ssim
    tb_log_dir = os.path.join('result', opt['dataset'], 'logs')             # save trained logs

    if not os.path.exists(save_weights_file):
        os.makedirs(save_weights_file)
    if not os.path.exists(save_images_file):
        os.makedirs(save_images_file)
    if not os.path.exists(save_csv_file):
        os.makedirs(save_csv_file)
    if not os.path.exists(tb_log_dir):
        os.makedirs(tb_log_dir)

    use_pretrain = True  # 如果需要使用checkpoint的参数，需考虑关闭scheduler
    pretrain_weights = os.path.join(save_weights_file, 'model_best.pth')

    root_path = "./freetech_dataset" 
    dataset = load_data_RawRGB_Paired(
        root_path=root_path,
        raw_width=1920, raw_height=1280,
        rgb_width=1920, rgb_height=1280,
        patch_size=512,
        training=True,
        normalize_range=(0, 1)  # 输出范围[0, 1]
    )
    
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 设置随机种子保证可重复性
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt['batch_size'],
        shuffle=True,  # 训练时打乱数据
        num_workers=4,  
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # 丢弃最后一个不完整的batch
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt['batch_size'],
        shuffle=False,  # 测试时不打乱数据
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False  # 测试时保留所有数据
    )

    # 新加：RepNR 参数
    num_virtual_cams = 5  # 虚拟相机数
    pretrain_mode = False  # RepNR模块的虚拟相机预训练
    pretrain_epochs = 50  # 预训 epochs (短)
    finetune_epochs = opt['epochs'] - pretrain_epochs  # 续训

    device = 'cuda'

    if opt['model_size'] == 'S':
        dim = 32
    elif opt['model_size'] == 'B':
        dim = 48
    else:
        dim = 64

    model = RawFormer(dim=dim,use_rep_nr=True,num_virtual_cams=num_virtual_cams,pretrain_mode=pretrain_mode)

    print('\nTrainable parameters : {}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('\nTotal parameters : {}\n'.format(sum(p.numel() for p in model.parameters())))
    model = model.to(device)
    print('Device on cuda: {}'.format(next(model.parameters()).is_cuda))

    start_epoch = 0
    end_epoch = opt['epochs']
    best_psnr = 0
    best_epoch = 0

    ######### Loss ###########
    loss_criterion = torch.nn.L1Loss()

    ######### Pretrain Mode ###########
    if not pretrain_mode:  # 微调: 冻结 IMNR (只训 OMNR/CSA)
        for name, module in model.named_modules():
            if 'imnr' in name:  # 假设 RepNR 内 IMNR
                for param in module.parameters():
                    param.requires_grad = False
        print("Finetune mode: Frozen IMNR, only train OMNR/CSA")

    ######### Scheduler ###########
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['base_lr'])
    if use_pretrain:
        checkpoint = torch.load(pretrain_weights)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        start_epoch = checkpoint['epoch'] + 1

    print("Using warmup and cosine strategy!")
    warmup_epochs = 20
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, end_epoch-warmup_epochs, eta_min=1e-5)
    # scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler = None  # 如果checkpoint的参数是使用了pretrain_mode得到的，推荐直接关闭scheduler

    torch.cuda.empty_cache()
    loss_scaler = torch.cuda.amp.GradScaler()    # 计算loss时用到的梯度scaler

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'valid_PSNR': 0,
        'worst_PSNR': 0,
        # 'valid_SSIM': 0,
        'best_PSNR': 0,
        'best_epoch': 0,
        'epoch_time': 0,
        'epoch_loss': 0,
        'epoch_LR': 0,
    }

    epoch = start_epoch
    while epoch < opt['epochs'] + 1:
        epoch_start_time = time.time()
        epoch_loss = 0

        for i, img in enumerate(tqdm.tqdm(train_loader)):
            optimizer.zero_grad()
            input_raw = img[0].to(device)
            gt_rgb = img[1].to(device)

            with torch.cuda.amp.autocast():
                batch_losses = []
                if pretrain_mode:
                    for cam_id in range(num_virtual_cams):
                        pred_rgb = model(input_raw, cam_id=cam_id)
                        pred_rgb = torch.clamp(pred_rgb, 0, 1)
                        loss = loss_criterion(pred_rgb, gt_rgb)
                        batch_losses.append(loss)
                    loss = torch.mean(torch.stack(batch_losses))
                else:  # 默认 cam_id=0
                    pred_rgb = model(input_raw, cam_id=0)
                    pred_rgb = torch.clamp(pred_rgb, 0, 1)
                    loss = loss_criterion(pred_rgb, gt_rgb)
                
            loss_scaler.scale(loss).backward()
            loss_scaler.step(optimizer)
            loss_scaler.update()
            epoch_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        #### Evaluation ####
        with torch.no_grad():
            model.eval()
            psnr_val_rgb = []
            worst_psnr = best_psnr
            for ii, data_val in enumerate(tqdm.tqdm(test_loader)):
                input_raw = data_val[0].to(device)
                gt_rgb = data_val[1].to(device)
                with torch.cuda.amp.autocast():
                    pred_rgb = model(input_raw, cam_id=0)
                pred_rgb = torch.clamp(pred_rgb, 0, 1)
                psnr_val_rgb.append(PSNR((data_val[1].numpy().transpose(0, 2, 3, 1)*255).astype(np.uint8),
                                         (pred_rgb.detach().cpu().numpy().transpose(0, 2, 3, 1)*255).astype(np.uint8)))

            # 平均 PSNR
            psnr_val_rgb_avg = sum(psnr_val_rgb) / len(test_loader) if psnr_val_rgb else 0.0

            # 当前 epoch 最低 PSNR
            worst_psnr = min(psnr_val_rgb) if psnr_val_rgb else float('inf')  # 防空列表

            if psnr_val_rgb_avg > best_psnr:
                best_psnr = psnr_val_rgb_avg
                best_epoch = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(save_weights_file, "model_best.pth"))

            print("------------------------------------------------------------------")
            print("[PSNR SID: %.4f, Worst_PSNR: %.4f] ----  [best_Ep_SID: %d, Best_PSNR_SID: %.4f] " % (psnr_val_rgb_avg, worst_psnr, best_epoch, best_psnr))
            model.train()

        if scheduler is not None:
            current_lr = scheduler.get_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']
        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,epoch_loss, current_lr))
        print("------------------------------------------------------------------")

        if writer_dict:
            writer = writer_dict['writer']
            writer.add_scalar('valid_PSNR', psnr_val_rgb_avg, epoch)
            writer.add_scalar('worst_PSNR', worst_psnr, epoch)
            writer.add_scalar('best_PSNR', best_psnr, epoch)
            writer.add_scalar('best_epoch', best_epoch, epoch)
            writer.add_scalar('epoch_time', time.time() - epoch_start_time, epoch)
            writer.add_scalar('epoch_loss', epoch_loss, epoch)
            writer.add_scalar('epoch_LR', current_lr, epoch)

        if epoch == end_epoch:
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(save_weights_file, "model_{}.pth".format(epoch)))

        if pretrain_mode and epoch == pretrain_epochs:  # 预训结束前保存
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'pretrain': True},
                       os.path.join(save_weights_file, "model_pretrained.pth"))
            print("Pretrain finished, saving checkpoint")

        if pretrain_mode and epoch >= pretrain_epochs:
            pretrain_mode = False
            model.pretrain_mode = False
            scheduler = None
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt['base_lr'] * 0.1
            for name, module in model.named_modules():
                if 'imnr' in name.lower():
                    for param in module.parameters():
                        param.requires_grad = False
            print(f"Switch to finetune at epoch {epoch + 1}")

        epoch += 1

    print("Now time is : ", datetime.datetime.now().isoformat())
    print('Model saved in: ', save_weights_file)