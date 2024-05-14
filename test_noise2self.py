import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
from scipy.io import loadmat
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset as dset
from torch.nn import MSELoss
from torch.optim import Adam

from dncnn import DnCNN
from mask import Masker

from multiprocessing import freeze_support

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    batch_size = 1
    model = DnCNN(1, num_of_layers=num_of_layers)
    # 학습된 모델 불러오기
    model_path = os.path.join(os.getcwd(), "trained_models/noise2self_checkerboard.pt")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device)
    model.eval()

    test_data_path = 'data/test_imgs.mat'
    test_dataset = mydataset(test_data_path)
    # numworkers=-1 되나?
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)

    result_path = "results/noise2self"
    total_test_rmse = 0

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            clean_images, noisy_images = batch
            noisy_images = noisy_images.to(device, dtype=torch.float)
            denoised_images = model(noisy_images)

            denoised_images_np = np.squeeze(denoised_images.cpu().numpy().astype(np.float64))
            clean_images_np = np.squeeze(clean_images.numpy().astype(np.float64))

            test_rmse = np.sqrt(mean_squared_error(denoised_images_np, clean_images_np))
            total_test_rmse += test_rmse

            print('test_rmse : ' + str(test_rmse))
        total_test_rmse = total_test_rmse / (idx + 1)
        # test결과 저장
        # np.savez(result_path, total_test_rmses=total_test_rmse)
        print("Average Test RMSE: %.4f" % total_test_rmse)


if __name__ == '__main__':
    freeze_support()

    main()