import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.io import loadmat
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset as dset
from torch.nn import MSELoss
from torch.optim import Adam
from multiprocessing import freeze_support

from dncnn import DnCNN
from mask import Masker
from tensorboard_logger import TensorBoardLogger



"""
기존에 있던 학습된 모델은 trained_models/에 있고
train 스크립트로 새로 학습하면 results/에 저장됨

results/{model_name}에는 tensorboard로 그래프 그리는데 필요한 값들이 저장됨
새로 train/test을 진행한다면 해당 디렉토리를 비우고나서 진행해야 그래프에 새로운 값들만 표시됨
"""



# np.set_printoptions(threshold=np.inf)
# torch.set_printoptions(threshold=float('inf'))

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"]="0"

set_seed(100)

result_path = "results/noise2self_donut"
ENABLE_TENSORBOARD = False

class mydataset(dset):
    def __init__(self, folderpath_img):
        super(mydataset, self).__init__()

        self.clean_images = loadmat(folderpath_img)["xtrue"].transpose(2, 0, 1).astype(np.float64)
        self.noisy_images = loadmat(folderpath_img)["xfbp"].transpose(2, 0, 1).astype(np.float64)

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, index):
        clean_images = np.expand_dims(self.clean_images[index], axis=0)
        noisy_images = np.expand_dims(self.noisy_images[index], axis=0)

        return (clean_images, noisy_images)


def main():
    num_of_layers = 8
    batch_size = 2
    lr = 1e-1
    step_size = 10
    gamma = 0.95

    num_epoch = 1000

    masker = Masker(width=4, mode='interpolate')

    model = DnCNN(1, num_of_layers=num_of_layers)
    model = model.to(device)


    loss_function_mean = MSELoss(reduction='mean')
    loss_function_sum = MSELoss(reduction='sum')

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_dataset = mydataset('data/train_imgs.mat')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)


    losses = []
    train_rmses = []
    logger = TensorBoardLogger(log_dir=result_path, enabled=ENABLE_TENSORBOARD)
    logger.launch_tensorboard()

    for epoch in range(num_epoch):
        model.train()
        total_loss = 0
        total_train_rmse = 0
        total_psnr = 0

        for idx, batch in enumerate(train_loader):
            clean_images, noisy_images = batch

            noisy_images = noisy_images.to(device, dtype=torch.float)
            
            net_input, mask = masker.mask(noisy_images, idx)

            net_output = model(net_input)

            denoised = model(noisy_images)

            loss = loss_function_sum(net_output * mask, noisy_images * mask) / (batch_size * torch.sum(mask))

            denoised = np.squeeze(denoised.detach().cpu().numpy().astype(np.float64))
            clean_image = np.squeeze(clean_images.cpu().numpy().astype(np.float64))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_rmse = np.sqrt(mean_squared_error(denoised, clean_image))
            train_psnr = psnr(denoised, clean_image, data_range=4095)

            total_train_rmse += train_rmse
            total_loss += loss.item()
            total_psnr += train_psnr

        scheduler.step()

        avg_train_loss = total_loss / (idx + 1)
        avg_train_rmse = total_train_rmse / (idx + 1)
        avg_train_pnsr = train_psnr / (idx + 1)

        losses.append(avg_train_loss)
        train_rmses.append(avg_train_rmse)

        torch.save(model.state_dict(), result_path + ".pt")
        np.savez(result_path, losses=losses, train_rmses=train_rmses)

        logger.log_scalar('Loss', avg_train_loss, epoch)
        logger.log_scalar('RMSE', avg_train_rmse, epoch)
        logger.log_scalar('PNSR', avg_train_pnsr, epoch)
        print("(", (epoch + 1), ") Training Loss: %.1f" % avg_train_loss, ", RMSE, : %.1f" % avg_train_rmse)

    logger.close()

if __name__ == '__main__':
    freeze_support()

    main()