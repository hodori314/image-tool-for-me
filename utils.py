import torch
import torch.nn as nn
import torchvision

import torchvision.transforms as transforms
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
import time

import lpips

batch_size = 1

transform = transforms.Compose(
    [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cifar10set = torchvision.datasets.CIFAR10(root='~/data/CIFAR10', train=True, download=False, transform=transform)
cifar10loader = torch.utils.data.DataLoader(cifar10set, batch_size=batch_size, shuffle=False, num_workers=0)

channel = 3
im_size = (32, 32)
num_classes = 2
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform=transforms.Compose([
                        transforms.Resize(im_size),
                        transforms.CenterCrop(im_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])
celebtrainset = torchvision.datasets.ImageFolder(root='/home/guest-yhj/data/celebA-condensed/gender', transform = transform)

selected_indices = [12, 17, 55, 46, 30, 15, 54, 45, 15, 22, 55, 53, 27, 5, 43, 36, 27, 14, 41, 42]
subset_dataset = torch.utils.data.Subset(celebtrainset, selected_indices)

celebAloader = torch.utils.data.DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def l2_norm(x, y):
    v = x - y
    return torch.norm(v, p=2).item()

loss_fn_lpips = lpips.LPIPS(net='alex')

def lpips_loss(img_batch, ref_batch):
    [B, C, m, n] = img_batch.shape
    lpips_losses = []
    for sample in range(B):
        lpips_losses.append(loss_fn_lpips(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
    lpips_loss = torch.stack(lpips_losses, dim=0).mean()

    return lpips_loss.item()

def imshow(img, file_name):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('%s.png'%(file_name))

def imshow_2(img1, img2, file_name, dist):
    img1 = img1 / 2 + 0.5
    npimg1 = img1.detach().numpy()
    img2 = img2 / 2 + 0.5
    npimg2 = img2.detach().numpy()

    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.title("original image")
    plt.axis('off')
    plt.imshow(np.transpose(npimg1, (1, 2, 0)))
    
    plt.subplot(1,2,2)
    plt.title("unpacked image %.3f"%(dist))
    plt.axis('off')
    plt.imshow(np.transpose(npimg2, (1, 2, 0)))

    plt.savefig('%s.png'%(file_name), bbox_inches='tight')

def image_distance_dataset(myloader):
    label_dist = defaultdict(float)

    for i, data in enumerate(myloader, 0):
        input, label = data
        label = label.item()
        
        print('distance for label%d at'%(label) , time.strftime('%Y-%m-%d %H:%M:%S'))

        candi_data = []
        candi_norm = []

        for i_origin, data_origin in enumerate(celebAloader, 0):
            input_origin, label_origin = data_origin
            label_origin = label_origin.item()
            if label_origin == 1:
                continue
            
            candi_data.append(data_origin)
            # candi_norm.append(lpips_loss(input_origin, input))
            candi_norm.append(l2_norm(input_origin, input))
        
        min_idx = torch.argmin(torch.tensor(candi_norm))
        min_input, min_label = candi_data[min_idx.item()]

        imshow_2(torchvision.utils.make_grid(min_input), torchvision.utils.make_grid(input), 
                'label%d_minlabel%d'%(label, min_label), candi_norm[min_idx.item()])
        
        label_dist[label] = candi_norm[min_idx.item()]

    print('gias image <--> most similar image in original set')
    # print('min label:', torch.argmin( torch.Tensor( list(label_dist.values())) ))
    # print('max label:', torch.argmin( torch.Tensor( list(label_dist.values())) ))

def psnr(img_batch, ref_batch, batched=False, factor=1.0):
    """Standard PSNR."""
    def get_psnr(img_in, img_ref):
        mse = ((img_in - img_ref)**2).mean()
        if mse > 0 and torch.isfinite(mse):
            return (10 * torch.log10(factor**2 / mse))
        elif not torch.isfinite(mse):
            return img_batch.new_tensor(float('nan'))
        else:
            return img_batch.new_tensor(float('inf'))

    if batched:
        psnr = get_psnr(img_batch.detach(), ref_batch)
    else:
        [B, C, m, n] = img_batch.shape
        psnrs = []
        for sample in range(B):
            psnrs.append(get_psnr(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
        psnr = torch.stack(psnrs, dim=0).mean()

    return psnr.item()

def mse(img_batch, ref_batch, batched=False, factor=1.0):
    """Standard PSNR."""
    def get_mse(img_in, img_ref):
        mse = ((img_in - img_ref)**2).mean()
        return mse

    if batched:
        psnr = get_mse(img_batch.detach(), ref_batch)
    else:
        [B, C, m, n] = img_batch.shape
        psnrs = []
        for sample in range(B):
            psnrs.append(get_mse(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
        psnr = torch.stack(psnrs, dim=0).mean()

    return psnr.item()

