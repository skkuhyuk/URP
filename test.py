import sys
import os
from multiprocessing import freeze_support

import numpy as np
import torch

import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.io import loadmat
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset as dset

from dncnn import DnCNN
from tensorboard_logger import TensorBoardLogger

#os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#test_data_path='data/test_imgs.mat'
test_data_path='C:/Users/JD/OneDrive/바탕 화면/urp/data/perfect_same_simul_test.mat'

# noise2self
#model_path = 'trained_models/ours_ssrl_noise2self_checkerboard.pt' # "trained_models/noise2self.pt"
#result_path = "results/noise2self"

# ssrl_noise2self
#model_path = 'results/ssrl_ex1/2024_05_03_01_18/ssrl_ex1_f_200.pt' # "trained_models/noise2self.pt"

# result_path = "results/ssrl_ex1/2024_05_06_9_13"
#model_path = "results/ssrl_ex1/2024_05_05_13_13/ssrl_ex1_g_200.pt"
# model_path = "results/noise2self_checkerboard/2024_05_05_00_22.pt"
# ssrl_ex1_f, ssrl_ex1_g
model_path = "4090trained/SSRL_N2Self/ssrl_noise2self_checkerboard_200.pt"

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
    batch_size = 1

    # train 할때와 동일한 layer 구조로 모델 생성해야함
    num_of_layers = 8
    model = DnCNN(1, num_of_layers=num_of_layers)
    # 학습된 모델 불러오기
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device)
    # 학습x
    model.eval()

    test_dataset = mydataset(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)


    avg_test_rmse = 0
    avg_test_psnr = 0
    #dir="C:/Users/JD/OneDrive/바탕 화면/urp/results/ssrl_ex1/2024_05_06_09_32/test_picture_g"
    dir="results/4090result/SSRL"
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            clean_images, noisy_images = batch

            save_picture(idx=idx,dir=dir,image=clean_images,imgname="clean")

            noisy_images = noisy_images.to(device, dtype=torch.float)

            save_picture(idx=idx,dir=dir,image=noisy_images,imgname="dirty")
            denoised_images = model(noisy_images)

            save_picture(idx=idx,dir=dir,image=denoised_images,imgname="denoised")
            
            denoised_images_np = np.squeeze(denoised_images.cpu().numpy().astype(np.float64))
            clean_images_np = np.squeeze(clean_images.numpy().astype(np.float64))
            test_rmse = np.sqrt(mean_squared_error(denoised_images_np, clean_images_np))
            test_psnr = psnr(denoised_images_np, clean_images_np, data_range=4095)

            avg_test_rmse += test_rmse
            avg_test_psnr += test_psnr

            print(f'{idx} test_rmse : {test_rmse}, {idx} test_psnr : {test_psnr}')

        avg_test_rmse = avg_test_rmse / (idx + 1)
        avg_test_psnr = avg_test_psnr / (idx + 1)

        # test결과 저장
        # np.savez(result_path, total_test_rmses=total_test_rmse)
        print("Average Test RMSE: %.4f" % avg_test_rmse)
        print("Average Test PSNR: %.4f" % avg_test_psnr)

        #if epoch % 10 == 0:
        #if epoch>=0:

               

def save_picture(idx, dir, image, imgname):
    image=image.cpu().squeeze()
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.imshow(image,vmin=800,vmax=1200,cmap='gray')
    plt.xlabel(f'{imgname}_{idx}')
    plt.savefig(fr"{dir}/{imgname}_{idx}.png")
if __name__ == '__main__':
    freeze_support()

    main()