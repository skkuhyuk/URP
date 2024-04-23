import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset as dset
from scipy.io import loadmat
from scipy import io
import torchvision.models as models
from simsiam import SimSiam
#1장만 가져와서 변화를 확인하는 코드입니다. 뭐가 문제인지 잘 모르겠습니다...
########### Masker & Hyperparam ###################
kernel = torch.Tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], (0.0, 1.0, 0.0)])
kernel_inv = torch.ones(kernel.shape) - kernel

kernel = kernel[np.newaxis, np.newaxis, :, :]
kernel = kernel / kernel.sum()

kernel_inv = kernel_inv[np.newaxis, np.newaxis, :, :]
kernel_inv = kernel_inv / kernel_inv.sum()
replicate_unit = torch.Tensor([[0.0, 1.0], [1.0, 0.0]])
S_mask = replicate_unit.repeat(256, 256)
S_mask_inv = torch.ones(S_mask.shape) - S_mask
optimizer=torch.optim.Adam(model.parameters(), lr=1e-1)
criterion = nn.CosineSimilarity(dim=1)
########### Masker & Hyperparam ###################


#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

########### Noisy만 필요하므로 Noisy만 불러옴 ###################
class mydataset(dset):
    def __init__(self, folderpath_img):
        super(mydataset, self).__init__()

        #self.clean_images = loadmat(folderpath_img)["xtrue"].transpose(2, 0, 1).astype(np.float64)
        self.noisy_images = loadmat(folderpath_img)["xfbp"].transpose(2, 0, 1).astype(np.float64)

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, index):
        #clean_images = np.expand_dims(self.clean_images[index], axis=0)
        noisy_images = np.expand_dims(self.noisy_images[index], axis=0)

        return (noisy_images)
########### Noisy만 필요하므로 Noisy만 불러옴 ###################

########### 0번째 사진만 불러옴 ###################
def load_image(image_path):
    image = image_path
    dirty = DataLoader(image, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
    for batch_idx, batch in enumerate(dirty):
        data = batch[0]
        break

    return data
########### 0번째 사진만 불러옴 ###################
epochs_num=100
model = SimSiam()

for epoch in range(epochs_num):

    model.train()
    #########데이터 불러오기&전처리##################
    image = mydataset('train_imgs.mat')
    train_loader = DataLoader(image, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
    dirty_image = load_image(image)
    m = torch.nn.ReflectionPad2d(1)
    dirty_image = dirty_image.to(dtype=torch.float)
    dirty_tensor = m(dirty_image)
    dirty_tensor=dirty_tensor.unsqueeze(0)
    filtered_tensor = torch.nn.functional.conv2d(dirty_tensor, kernel, stride=1, padding=0)
    #########데이터 불러오기&전처리##################

    #dirty_image.shape=1,1,512
    #dirty_tensor.shape=1,1,512,512
    #filtered_tensor.shape=1,1,512,512
    #xj.shape=1,1,512,512

    xj = S_mask * filtered_tensor + S_mask_inv * dirty_image
    xjc = S_mask_inv * filtered_tensor + S_mask * dirty_image
    ####################################################################
    #Error: ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 2048])
    #왜 서식이 안 맞는 거죠???
    p1, p2, z1, z2 = model(xj, xjc)
    ####################################################################
    loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

    print("(", (epoch), ") Training Loss: %.2f" %loss )
    print("Predictor 1 output :", p1)
    print("Predictor 1 output :", p2)
    print("Predictor 1 output :", z1)
    print("Predictor 1 output :", z2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#print("Predictor 1 output shape:", p1.shape)
#print("Predictor 2 output shape:", p2.shape)
#print("Feature 1 shape:", z1.shape)
#print("Feature 2 shape:", z2.shape) 1,2048
