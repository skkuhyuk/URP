import sys
import os
import time
import datetime
import json

import numpy as np
import torch
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
from tensorboard_logger import TensorBoardLogger

""" ---ex1---
1. f, g 둘 다 처음부터 학습함
2. f는 기존 ssrl처럼 1/2 * (mse(f(Xj), g(Xjc)) + mse(f(Xjc), g(Xj)))
3. g는 noise2self에서와 동일하게 학습
-g는 f랑 독립적으로 학습돼서 기존 noise2self랑 완전히 똑같이 학습 될듯(아마도?)
-rmse같은 변수 앞에 f_ g_ 안달려있으면 모두 f기준임
-g는 어차피 noise2self대로 학습될거니까 g관련 로깅은 모두 validation 용도,, 
"""

"""
기존에 있던 학습된 모델은 trained_models/에 있고
train 스크립트로 새로 학습하면 results/에 저장됨

results/{time}/{model_name}에는 tensorboard로 그래프 그리는데 필요한 값들이 저장됨
새로 train/test을 진행한다면 해당 디렉토리를 비우고나서 진행해야 그래프에 새로운 값들만 표시됨
"""


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

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

def save_picture(idx, dir, image, imgname):
    image=image.squeeze()
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.imshow(image,vmin=800,vmax=1200,cmap='gray')
    plt.xlabel(f'{imgname}_{idx}')
    plt.savefig(fr"{dir}/{imgname}_{idx}.png")
def main():
    torch.autograd.set_detect_anomaly(True)

    ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

    set_seed(100)
    time_now=time.time()
    formatted_time = datetime.datetime.fromtimestamp(time_now).strftime('%Y_%m_%d_%H_%M')

    model_name = "ssrl_ex1"
    result_path = "results/" + model_name + '/' + formatted_time

    ENABLE_LOGGING = True
    RUN_TENSORBOARD = False     # True : 브라우저에서 localhost:6006 접속

    # logger = TensorBoardLogger(log_dir=result_path, logging_enabled=ENABLE_LOGGING, run=RUN_TENSORBOARD)
    # logger.launch_tensorboard()
    ###

    num_of_layers = 8
    batch_size = 2
    lr = 1e-2
    step_size = 10 
    gamma = 0.95
    epoch = 0 # 이어서 학습하는경우의 시작할 epoch # results/model_name에 있는 .pt 파일 참고
    num_epoch = 200
    
    config = {
        "description" : "중요한 실험이면 여기에 설명 작성해주세요",
        "num_of_layers": num_of_layers,
        "batch_size": batch_size,
        "learning_rate": lr,
        "step_size": step_size,
        "gamma": gamma,
        "num_epoch": num_epoch
    }
    with open(f'{result_path}/config.json', 'w') as f:
        json.dump(config, f, indent=4)


    f_losses = []
    g_losses = []
    f_train_rmses = []
    g_train_rmses = []
    f_psnrs=[]
    g_psnrs=[]
    # 학습하던 모델의 가중치와 loss, rmse 불러옴
    f_trained_model = f"{result_path}_f_{epoch}.pt"
    g_trained_model = f"{result_path}_g_{epoch}.pt"
    metrics_path = result_path + f"/{model_name}_{epoch}"
    model_fx = DnCNN(1, num_of_layers=num_of_layers)
    model_gx = DnCNN(1, num_of_layers=num_of_layers)
    if os.path.exists(f_trained_model) and os.path.exists(g_trained_model) and os.path.exists(f"{metrics_path}.npz"):
        model_fx.load_state_dict(torch.load(f_trained_model))
        model_gx.load_state_dict(torch.load(g_trained_model))
        metrics = np.load(f"{metrics_path}.npz")
        f_losses = metrics['f_losses'].tolist()
        g_losses = metrics['g_losses'].tolist()
        f_train_rmses = metrics['f_train_rmses'].tolist()
        g_train_rmses = metrics['g_train_rmses'].tolist()
        print(f"epoch {epoch + 1}부터 이어서 학습을 시작합니다")
    else:
        epoch = 0
        print("처음부터 학습을 시작합니다")

    start_time=time.time()
    model_fx = model_fx.to(device)
    model_gx = model_gx.to(device)

    
    model_fx.train()
    model_gx.train()


    loss_function = MSELoss()
    loss_function_sum = MSELoss(reduction='sum')

    f_optimizer = Adam(model_fx.parameters(), lr=lr)
    g_optimizer = Adam(model_gx.parameters(), lr=lr)
    f_scheduler = torch.optim.lr_scheduler.StepLR(f_optimizer, step_size=step_size, gamma=gamma)
    g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=step_size, gamma=gamma)

    train_dataset = mydataset('data/perfect_same_simul_train.mat')
    #train_dataset = mydataset('data/train_imgs.mat')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)

    kernel = torch.Tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], (0.0, 1.0, 0.0)])
    kernel_inv = torch.ones(kernel.shape) - kernel

    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = kernel / kernel.sum()

    kernel_inv = kernel_inv[np.newaxis, np.newaxis, :, :]
    kernel_inv = kernel_inv / kernel_inv.sum()

    kernel = kernel.to(device)
    kernel_inv = kernel_inv.to(device)

    replicate_unit = torch.Tensor([[0.0, 1.0], [1.0, 0.0]])
    S_mask = replicate_unit.repeat(256, 256)
    S_mask = S_mask.to(device)
    S_mask_inv = torch.ones(S_mask.shape).to(device) - S_mask
    

    # reflection padding
    m = torch.nn.ReflectionPad2d(1)

    log_clean_images = None
    log_noisy_images = None
    log_denoised_images = None
    for epoch in range(epoch + 1, num_epoch + 1):
        
        f_total_loss = 0
        g_total_loss = 0
        f_total_train_rmse = 0
        g_total_train_rmse = 0
        f_total_psnr=0
        g_total_psnr=0
        epoch_train_time = time.time()
        for idx, batch in enumerate(train_loader):
            index_start=time.time()
            clean_images, noisy_images = batch
            clean_image = np.squeeze(clean_images.cpu().numpy().astype(np.float64))
        
            # noisy_images_log = noisy_images
            noisy_images = noisy_images.to(device, dtype=torch.float)

            # reflection padding
            noisy_images_padded = m(noisy_images)

            # interpolation
            filtered_tensor = torch.nn.functional.conv2d(noisy_images_padded, kernel, stride=1, padding=0)
            # 이게 Xj면
            net_input1 = filtered_tensor * S_mask + noisy_images * S_mask_inv
            # 이건 Xjc
            net_input2 = filtered_tensor * S_mask_inv + noisy_images * S_mask

            # f(Xj)
            net_output_fx1 = model_fx(net_input1)
            # f(Xjc)
            net_output_fx2 = model_fx(net_input2)

            # g(Xj)
            net_output_gx1 = model_gx(net_input1)
            # g(Xjc)
            net_output_gx2 = model_gx(net_input2)

            # with torch.no_grad():
            g_loss1 = loss_function_sum(net_output_gx1 * S_mask, noisy_images * S_mask) / (batch_size * torch.sum(S_mask))
            #S_mask-> input & output 
            g_loss2 = loss_function_sum(net_output_gx2 * S_mask_inv, noisy_images * S_mask_inv) / (batch_size * torch.sum(S_mask_inv))
            g_loss = (g_loss1 + g_loss2) / 2.0

            with torch.no_grad():
                f_denoised = model_fx(noisy_images)
                g_denoised = model_gx(noisy_images)
            # denoised_output = denoised
            f_denoised = np.squeeze(f_denoised.detach().cpu().numpy().astype(np.float64))
            g_denoised = np.squeeze(g_denoised.detach().cpu().numpy().astype(np.float64))
            
            f_train_rmse = np.sqrt(mean_squared_error(f_denoised, clean_image))
            g_train_rmse = np.sqrt(mean_squared_error(g_denoised, clean_image))
            f_train_psnr = psnr(f_denoised, clean_image, data_range=4095)
            g_train_psnr = psnr(g_denoised, clean_image, data_range=4095)\
            
            model_gx.zero_grad()  
            g_loss.backward(retain_graph=True)              # g 학습
            g_optimizer.step()


            # mse(f(Xj), g(Xjc))
            f_loss1 = loss_function(net_output_fx1, net_output_gx2.detach())
            # mse(f(Xjc), g(Xj))
            f_loss2 = loss_function(net_output_fx2, net_output_gx1.detach())
            f_loss = (f_loss1 + f_loss2) / 2.0
            
            # model_gx.requires_grad_(False) # g gradient 비활성화
            model_fx.zero_grad()
            f_loss.backward()              # f 학습
            f_optimizer.step()
            # model_gx.requires_grad_(True)  # g gradient 다시 활성화
        

            f_total_train_rmse += f_train_rmse
            g_total_train_rmse += g_train_rmse
            f_total_loss += f_loss.item()
            g_total_loss += g_loss.item()
            f_total_psnr += f_train_psnr
            g_total_psnr += g_train_psnr
            if idx % 10 == 0:
                print("idx : " + str(idx)) 

        f_scheduler.step()
        g_scheduler.step()
        
        avg_f_train_loss = f_total_loss / (idx + 1)
        avg_g_train_loss = g_total_loss / (idx + 1)
        avg_f_train_rmse = f_total_train_rmse / (idx + 1)
        avg_g_train_rmse = g_total_train_rmse / (idx + 1)
        avg_f_train_pnsr = f_total_psnr / (idx + 1)
        avg_g_train_pnsr = g_total_psnr / (idx + 1)

        f_losses.append(avg_f_train_loss)
        g_losses.append(avg_g_train_loss)
        f_train_rmses.append(avg_f_train_rmse)
        g_train_rmses.append(avg_g_train_rmse)
        f_psnrs.append(avg_f_train_pnsr)
        g_psnrs.append(avg_g_train_pnsr)
        
        cur_time=time.time()
        time_elapsed=cur_time-start_time
        
        
        #print("Cal time: ", cal_time)
        if epoch % 100 == 0:
            torch.save(model_fx.state_dict(), result_path + f"/{model_name}_f_{epoch}.pt")
            torch.save(model_gx.state_dict(), result_path + f"/{model_name}_g_{epoch}.pt")
            np.savez(result_path + f"/{model_name}_{epoch}", f_losses=f_losses,g_losses=g_losses, f_train_rmses=f_train_rmses,  g_train_rmses=g_train_rmses, 
                     time_elapsed=time_elapsed, f_psnrs=f_psnrs,g_psnrs=g_psnrs)

        """
        if(epoch==1):
            save_picture(idx=epoch,dir=result_path,image=first_clean_image,imgname="clean")
            save_picture(idx=epoch,dir=result_path,image=first_noisy_image1,imgname="noisy")
        if(epoch%3==0):
            with torch.no_grad():
                denoised=model_fx(first_noisy_image)
                denoised2=model_gx(first_noisy_image)
                save_picture(idx=epoch,dir=result_path,image=denoised.cpu(),imgname="denoisedfx")
                save_picture(idx=epoch,dir=result_path,image=denoised.cpu(),imgname="denoisedgx")
        """

        logger.log_scalar('f train Loss', avg_f_train_loss, epoch)
        logger.log_scalar('g train Loss', avg_g_train_loss, epoch)
        # logger.log_scalar('RMSE', avg_train_rmse, epoch)
        logger.log_scalar('F PNSR', avg_f_train_pnsr, epoch)
        logger.log_scalar('G PNSR', avg_g_train_pnsr, epoch)
        print("(", (epoch), ") Training F Loss: %.1f" % avg_f_train_loss, ", Training G Loss: %.1f" % avg_g_train_loss, 
              ", F RMSE, : %.1f" % avg_f_train_rmse,", G RMSE, : %.1f" % avg_g_train_rmse, ", Time elapsed(sec): %.1f" % time_elapsed, 
              ", Training F Psnr; %.1f" %avg_f_train_pnsr, ", Training G Psnr; %.1f" %avg_g_train_pnsr )
        
    logger.close()


if __name__ == '__main__':
    freeze_support()

    main()