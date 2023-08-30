import subprocess
from tqdm import tqdm
from collections import OrderedDict
import torch
import time
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import collections
from sklearn.cluster import KMeans

from utils.dataset_utils import TrainDataset

from data_RGB import DataLoaderTrain, DataLoaderTest
import torch.nn.functional as F

from option import options as opt

from losses import ssim_loss
from AMIRNet import AMIRNet

from utils.val_utils import AverageMeter, compute_psnr_ssim

def extract_features(model, data_loader, pos, layers, pseudo_mask):
    model.eval()
    out_features = None
    loop_num = 5
    with torch.no_grad():
        for _ in range(loop_num):
            features = None
            for i,(degraded_img, clean_img, label, index) in enumerate(data_loader):
                #print(index)
                index = index.numpy().tolist()
                degraded_img = degraded_img.cuda()
                clean_img = clean_img.cuda()
                _,feature_out = model.module.resencoder(degraded_img, pos)
                _,gt_feature_out = model.module.resencoder(clean_img, pos)
                one_hot = None
                temp_feature_out = None
                for idx in range(pos):
                    temp_one_hot = F.one_hot(pseudo_mask[index,idx].long(),num_classes=layers[idx])
                    #one_hot = torch.cat((one_hot,temp_one_hot),dim=1) if one_hot is not None else temp_one_hot
                    sub_feature_out = (feature_out[idx].data.cpu() - gt_feature_out[idx].data.cpu()) * temp_one_hot.data.cpu()
                    temp_feature_out = torch.cat((temp_feature_out,sub_feature_out),dim=1) if temp_feature_out is not None else sub_feature_out
                features = torch.cat((features, temp_feature_out),dim=0) if features is not None else temp_feature_out
            out_features = out_features + features if out_features is not None else features
    model.train()
    return out_features / loop_num
def test(net, test_dataset, pos):
    testloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=3, drop_last=False, pin_memory=True)
    psnr = AverageMeter()
    ssim = AverageMeter()
    with torch.no_grad():
        for (degrad_patch, clean_patch, label, filename) in testloader:
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            restored,_ = net(degrad_patch, pos=pos)
            restored = torch.clamp(restored,0,1)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            #save_image_tensor(restored, output_path + degraded_name[0] + '.png')

        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))



if __name__ == '__main__':
    
    ####   set random seed 
    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    
    
    subprocess.check_output(['mkdir', '-p', opt.ckpt_path])

    trainset = DataLoaderTrain(rgb_dir=opt.data_path, patch_size=opt.patch_size)
    
    trainsetclass = [0] * trainset.sizex
    pseudo_mask = torch.zeros(trainset.sizex, 4)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    
    testset= DataLoaderTest(rgb_dir=opt.data_path, patch_size=None)
    
    # Network Construction
    start_epoch = 0
    device_ids = [i for i in range(torch.cuda.device_count())]
    leaf_num = 1
    cluster_layers = [1,2,2,2]
    ptr = 1
    softmax_list = [0]
    
    model = AMIRNet()
    model = model.cuda()
    
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.999),eps=1e-8)
    if opt.resume:
        ckpt = torch.load(opt.resume_path, map_location=torch.device(opt.cuda))
        state_dict = ckpt['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.find('module')!=-1:
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(ckpt['optimizer'])
        pseudo_mask = ckpt['pseudo_mask']
        start_epoch = ckpt['epoch'] + 1
        ptr = ckpt['ptr']
        for _ in range(start_epoch+1):
            scheduler.step()

    model = nn.DataParallel(model)
    model.train()
    
    # Optimizer and Loss
    CE = nn.CrossEntropyLoss().cuda()
    l1 = nn.L1Loss().cuda()
    smoothL1 = nn.SmoothL1Loss(beta=0.03).cuda()
    
    # Start training
    print('Start training...')
    for epoch in range(start_epoch, opt.epochs + 1):
        epoch_start_time = time.time()
        
        crossentropy_mean = 0
        restore_mean = 0
        cnt = 0
        torch.cuda.empty_cache()

        if epoch % opt.frequency_clustering == 0:
            num_cluster = cluster_layers[int(epoch / opt.frequency_clustering)]
            clusterloader = DataLoader(trainset, batch_size=32, pin_memory=True, shuffle=False,
                                drop_last=False, num_workers=opt.num_workers)
            if ptr == 0:
                features = extract_features(model, clusterloader,ptr+1,[1,2,4,8],pseudo_mask)
            else:
                features = extract_features(model, clusterloader,ptr,[1,2,4,8],pseudo_mask)
            idx_lists = [[] for _ in range(leaf_num)]
            for idx in range(len(trainsetclass)):
                idx_lists[trainsetclass[idx]].append(idx)
            sum_cluster = 0
            for idx_list in idx_lists:
                cluster = KMeans(n_clusters = num_cluster, random_state = 2*num_cluster)
                sub_features = features[idx_list,:]
                pseudo_labels_tmp = cluster.fit_predict(sub_features)
                for idx in range(len(idx_list)):
                    trainsetclass[idx_list[idx]] = sum_cluster + pseudo_labels_tmp[idx]
                    pseudo_mask[idx_list[idx]][ptr] = sum_cluster + pseudo_labels_tmp[idx]
                sum_cluster += num_cluster
            #softmax_list.append(ptr)
            leaf_num = sum_cluster
            ptr = ptr + 1
            #ptr += sum_cluster
        for(degraded_img, clean_img, label, img_index) in trainloader:
            degraded_img, clean_img = degraded_img.cuda(), clean_img.cuda()
            temp_pseudo_mask = pseudo_mask[img_index,:].cuda().long()
            restored, (mask, pred) = model(degraded_img, ptr, temp_pseudo_mask)
            restored = torch.clamp(restored,0,1)
            CE_mask_loss = 0
            for idx in range(ptr):
                CE_mask_loss = CE_mask_loss + CE(mask[idx], temp_pseudo_mask[:,idx])
            
            alpha = 0.85
            l1loss = smoothL1(restored, clean_img)
            ssimloss = ssim_loss(restored, clean_img)
            restore_loss = l1loss * alpha + (1 - alpha) * ssimloss
            contrast_loss = 0
            loss = restore_loss + CE_mask_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            crossentropy_mean += CE_mask_loss.item()
            restore_mean += restore_loss.item()
            cnt += 1
        scheduler.step()
        print('Epoch (%d)  Loss: restore_loss:%0.4f contrast_loss:%0.4f   time: %0.4f\n' % (
                    epoch, restore_mean / cnt, crossentropy_mean / cnt, time.time()-epoch_start_time
                ), '\r', end='')
        if epoch % opt.save_frequency == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                "pseudo_mask": pseudo_mask,
                "ptr": ptr
            }
            torch.save(checkpoint, opt.ckpt_path + 'epoch_' + str(epoch) + '.pth')
        if epoch % opt.test_frequency == 0:
            model.eval()
            test(model, testset, ptr)
            model.train()
