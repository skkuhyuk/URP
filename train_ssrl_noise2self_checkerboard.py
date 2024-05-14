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
# import wandb

from multiprocessing import freeze_support

from dncnn import DnCNN
#from tensorboard_logger import TensorBoardLogger
from train_noise2self_checkerboard import semi_result

"""
기존에 있던 학습된 모델은 trained_models/에 있고
train 스크립트로 새로 학습하면 results/에 저장됨

results/{model_name}에는 tensorboard로 그래프 그리는데 필요한 값들이 저장됨
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

    ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(device)
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"
    sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

    set_seed(100)

    time_now=time.time()
    formatted_time = datetime.datetime.fromtimestamp(time_now).strftime('%Y_%m_%d_%H_%M')

    model_name = "ssrl_noise2self_checkerboard"
    result_path = "results/" + model_name + '/' + formatted_time
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    ENABLE_LOGGING = True
    RUN_TENSORBOARD = False     # True : 브라우저에서 localhost:6006 접속

    #logger = TensorBoardLogger(log_dir=result_path, logging_enabled=ENABLE_LOGGING, run=RUN_TENSORBOARD)
    #logger.launch_tensorboard()
    ###



    num_of_layers = 8
    batch_size = 1
    lr = 1e-2
    step_size = 10
    gamma = 0.95
    epoch = 0 # 이어서 학습하는경우의 시작할 epoch # results/model_name에 있는 .pt 파일 참고
    num_epoch = 200

    config = {
        "description" : "train f only, 1~200",
        "num_of_layers": num_of_layers,
        "batch_size": batch_size,
        "learning_rate": lr,
        "step_size": step_size,
        "gamma": gamma,
        "num_epoch": num_epoch
    }
    with open(f'{result_path}/config.json', 'w') as f:
        json.dump(config, f, indent=4)

    losses = []
    train_rmses = []
    pnsrs = []

    # 학습하던 모델의 가중치와 loss, rmse 불러옴
    trained_model_path = result_path + f"/{model_name}_{epoch}"
    model = DnCNN(1, num_of_layers=num_of_layers)
    if os.path.exists(f"{trained_model_path}.pt") and os.path.exists(f"{trained_model_path}.npz"):
        model.load_state_dict(torch.load(result_path + f"/{model_name}_{epoch}.pt"))
        metrics = np.load(f"{trained_model_path}.npz")
        losses = metrics['losses'].tolist()
        train_rmses = metrics['train_rmses'].tolist()
        print(f"epoch {epoch + 1}부터 이어서 학습을 시작합니다")
    else:
        epoch = 0
        print("처음부터 학습을 시작합니다")

    model_gx = DnCNN(1, num_of_layers=num_of_layers)
    file_path = os.path.join(os.getcwd(), "results/noise2self_checkerboard/2024_05_05_00_22.pt")
    model_gx.load_state_dict(torch.load(file_path, map_location='cpu'))

    model = model.to(device)
    model_gx = model_gx.to(device)
    print("Good")
    
    model.train()
    model_gx.eval()
    # gx를 학습안시킨다는뜻
    # 이거 없애면 g도 학습시킨다는거니까 f랑 g둘다 동시에 학습하게됨 -> 이러면 학습이 제대로 안될거(아마도)
    


    loss_function = MSELoss()

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_dataset = mydataset('data/perfect_same_simul_train.mat')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    first_clean_image = train_dataset.clean_images[0].reshape(1, 1, 512, 512)
    first_noisy_image1 = train_dataset.noisy_images[0].reshape(1, 1, 512, 512)
    first_noisy_image = torch.from_numpy(first_noisy_image1).to(device, dtype=torch.float)

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
    start_time=time.time()
    for epoch in range(epoch + 1, num_epoch + 1):
        total_loss = 0
        total_train_rmse = 0
        total_psnr = 0
        epoch_train_time = time.time()
        for idx, batch in enumerate(train_loader):
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
            net_output1 = model(net_input1)
            # f(Xjc)
            net_output2 = model(net_input2)

            with torch.no_grad():
                # g(Xj)
                net_output_gx1 = model_gx(net_input1)
                # g(Xjc)
                net_output_gx2 = model_gx(net_input2)

            # mse(f(Xj), g(Xjc))
            loss1 = loss_function(net_output1, net_output_gx2)
            # mse(f(Xjc), g(Xj))
            loss2 = loss_function(net_output2, net_output_gx1)
            loss = (loss1 + loss2) / 2.0

            denoised = model(noisy_images)
            # denoised_output = denoised
            denoised = np.squeeze(denoised.detach().cpu().numpy().astype(np.float64))

            train_rmse = np.sqrt(mean_squared_error(denoised, clean_image))
            train_psnr = psnr(denoised, clean_image, data_range=4095)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_rmse += train_rmse
            total_loss += loss.item()
            total_psnr += train_psnr
            if idx % 10 == 0:
                print("idx : " + str(idx))            

            #print("idx : " + str(idx))
        epoch_train_time = time.time() - epoch_train_time
        print("epoch time : ", epoch_train_time)  

        scheduler.step()

        avg_train_loss = total_loss / (idx + 1)
        avg_train_rmse = total_train_rmse / (idx + 1)
        avg_train_psnr = total_psnr / (idx + 1)

        losses.append(avg_train_loss)
        train_rmses.append(avg_train_rmse)
        pnsrs.append(avg_train_psnr)
        
        cur_time=time.time()
        time_elapsed=cur_time-start_time
        if epoch % 50 == 0:
            torch.save(model.state_dict(), result_path + f"/{model_name}_{epoch}.pt")
            np.savez(result_path + f"/{model_name}_{epoch}", losses=losses, train_rmses=train_rmses, pnsrs=pnsrs, time_elapsed=time_elapsed)
        if(epoch==1):
            save_picture(idx=epoch,dir=result_path,image=first_clean_image,imgname="clean")
            save_picture(idx=epoch,dir=result_path,image=first_noisy_image1,imgname="noisy")
        if(epoch%3==0):
            with torch.no_grad():
                denoised=model(first_noisy_image)
                save_picture(idx=epoch,dir=result_path,image=denoised.cpu(),imgname="denoised")


        #logger.log_scalar('Loss', avg_train_loss, epoch)
        #logger.log_scalar('RMSE', avg_train_rmse, epoch)
        #logger.log_scalar('PSNR', avg_train_psnr, epoch)
        print("(", (epoch), ") Training Loss: %.1f" % avg_train_loss, ", RMSE, : %.1f" % avg_train_rmse, ", PSNR, : %.1f" % avg_train_psnr,
               ", Time elapsed(sec): %.1f" % time_elapsed)
    
    #logger.close()


if __name__ == '__main__':
    freeze_support()

    main()