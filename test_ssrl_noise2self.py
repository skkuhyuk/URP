import sys
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
from scipy.io import loadmat
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset as dset
# import wandb

from multiprocessing import freeze_support

from dncnn import DnCNN

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
    modelpath = os.path.join(os.getcwd(), "trained_models/ours_ssrl_noise2self.pt")
    model.load_state_dict(torch.load(modelpath, map_location='cpu'))

    model = model.to(device)
    model.eval()

    test_dataset = mydataset('data/test_imgs.mat')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)

    result_path = "results/ours_ssrl_noise2self"
    total_test_rmse = 0

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            clean_images, noisy_images = batch
            noisy_images = noisy_images.to(device, dtype=torch.float)
            denoised_images = model(noisy_images)

            denoised_images_np = denoised_images.cpu().numpy().squeeze()
            clean_images_np = clean_images.numpy().squeeze()

            test_rmse = np.sqrt(mean_squared_error(denoised_images_np, clean_images_np))
            total_test_rmse += test_rmse
            print('test_rmse : ' + str(test_rmse))
        average_rmse = total_test_rmse / len(test_loader)
        print(f'Average RMSE: {average_rmse:.4f}')

if __name__ == '__main__':
    freeze_support()

    main()