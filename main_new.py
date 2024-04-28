import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset as dset
from scipy.io import loadmat
from scipy import io
import torchvision.models as models
from torchvision.transforms import CenterCrop
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimSiam(base_encoder=models.resnet50).to(device)
pt_save_path = "/content/results.pt"
npz_save_path = "/content/epoch_data.npz"
results=[]
save_interval=1
import random
########### Masker & Hyperparam ###################
kernel = torch.Tensor([
    [[0.0, 1.0, 0.0],
     [1.0, 0.0, 1.0],
     [0.0, 1.0, 0.0]],
    [[0.0, 1.0, 0.0],
     [1.0, 0.0, 1.0],
     [0.0, 1.0, 0.0]],
    [[0.0, 1.0, 0.0],
     [1.0, 0.0, 1.0],
     [0.0, 1.0, 0.0]]
])

kernel_inv = torch.ones(kernel.shape) - kernel

kernel = kernel[np.newaxis, np.newaxis, :, :]
kernel = kernel / kernel.sum()

kernel_inv = kernel_inv[np.newaxis, np.newaxis, :, :]
kernel_inv = kernel_inv / kernel_inv.sum()
replicate_unit = torch.Tensor([[0.0, 1.0], [1.0, 0.0]])
S_mask = replicate_unit.repeat(112, 112)
S_mask_inv = torch.ones(S_mask.shape) - S_mask
optimizer=torch.optim.Adam(model.parameters(), lr=1e-1)
criterion = nn.CosineSimilarity(dim=1)
########### Masker & Hyperparam ###################


#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

########### Noisy만 필요하므로 Noisy만 불러옴 ###################
class mydataset(dset):
    def __init__(self, folderpath_img, transform=None):
        super(mydataset, self).__init__()
        self.transform=transform
        self.clean_images = loadmat(folderpath_img)["xtrue"].transpose(2, 0, 1).astype(np.float64)
        self.noisy_images = loadmat(folderpath_img)["xfbp"].transpose(2, 0, 1).astype(np.float64)

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, index):
        clean_images = np.expand_dims(self.clean_images[index], axis=0)
        noisy_images = np.expand_dims(self.noisy_images[index], axis=0)

        clean_images = np.transpose(clean_images, (1, 2, 0))
        noisy_images = np.transpose(noisy_images, (1, 2, 0))

        if self.transform:
            clean_images = self.transform(clean_images)
            noisy_images = self.transform(noisy_images)


        return (noisy_images)
########### Noisy만 필요하므로 Noisy만 불러옴 ###################

########### 0번째 사진만 불러옴 ###################
def load_image(image_path):
    image = image_path
    dirty = DataLoader(image, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
    print(dirty)

    for batch_idx, batch in enumerate(dirty):
        data = batch[0]
        break
    print(data.shape)
    return data
def load_image_jpg(image_path):
    image = Image.open(image_path).convert("RGB")  # JPEG 이미지를 RGB로 변환하여 불러옴
    return image

########### 0번째 사진만 불러옴 ###################
epochs_num=100
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

gaussian_blur = GaussianBlur(sigma=[0.1, 2.0])
for epoch in range(epochs_num):
    #model.train()
    #2장 이상 쓴다면 train으로
    model.eval()
    #########데이터 불러오기&전처리##################
    #image = mydataset("train_imgs.mat",transform=transform)
    dirty_image = load_image_jpg("image.jpg")
    dirty_image = dirty_image.convert("RGB")  # 이미지를 RGB 모드로 변환
    more_dirty_image = gaussian_blur(dirty_image)

    dirty_image = transform(dirty_image)
    more_dirty_image = transform(more_dirty_image)

    dirty_image = dirty_image.to(dtype=torch.float)
    more_dirty_image = more_dirty_image.to(dtype=torch.float)


    plt.imshow(more_dirty_image.permute(1, 2, 0))
    plt.show()
    print("MDI",more_dirty_image)
    plt.imshow(dirty_image.permute(1, 2, 0))  # 이미지 출력을 위해 차원을 변경하여 출력
    plt.show()
    print("DI",dirty_image)

    m = torch.nn.ReflectionPad2d(1)
    dirty_tensor1 = m(dirty_image.unsqueeze(0))
    filtered_tensor1 = torch.nn.functional.conv3d(dirty_tensor1, kernel, stride=1, padding=0)
    print("Filtered_tensor: ", filtered_tensor1)
    #########데이터 불러오기&전처리##################

    xj = S_mask * filtered_tensor1 + S_mask_inv * dirty_image
    xjc = S_mask_inv * filtered_tensor1 + S_mask * dirty_image
    xj = xj.view(1, 3, 224, 224).to(device)
    xjc = xjc.view(1, 3, 224, 224).to(device)
    print("Xj:",xj)
    print("Xjc:",xjc)
    ####################################################################
    p1, p2, z1, z2 = model(xj, xjc)
    ####################################################################
    loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

    print("(", epoch, ") Training Loss: %.2f" % loss)
    print("Predictor 1 output:", p1)
    print("Predictor 2 output:", p2)
    print("Feature 1:", z1)
    print("Feature 2:", z2)
    if epoch % save_interval == 0:
        epoch_result = {
            'epoch': epoch,
            'loss': loss.item(),
            'p1': p1.detach().cpu().numpy(),
            'p2': p2.detach().cpu().numpy(),
            'z1': z1.detach().cpu().numpy(),
            'z2': z2.detach().cpu().numpy(),
            'xj': xj.detach().cpu().numpy(),
            'xjc': xjc.detach().cpu().numpy()
        }
        results.append(epoch_result)
        torch.save(results, pt_save_path)
        epoch_data = {str(i): results[i] for i in range(len(results))}
        np.savez(npz_save_path, **epoch_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
