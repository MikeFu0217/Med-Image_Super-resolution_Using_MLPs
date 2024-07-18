import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils import dataloader3D, criterion
import os
from torch.utils.tensorboard import SummaryWriter

def readImg3D(img_dir):
    img3D = []
    img_names = os.listdir(img_dir)
    # img_names.sort(key=lambda x: int(x[:-4]))
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        img = plt.imread(img_path)
        img = img if (img.shape[-1]!=3 and img.shape[-1]!=4) else img.mean(axis=2)
        img3D.append(img)
    return np.array(img3D)

def saveImg3D(save_dir, name, img3D):
    save_dir = os.path.join(save_dir, name)
    os.makedirs(save_dir, exist_ok=True)
    for i in range(img3D.shape[0]):
        img = img3D[i, :, :]
        img_save_path = os.path.join(save_dir, f"{i}.jpg")
        plt.imsave(img_save_path, img, cmap="gray")

def train_and_evaluate(img_dir, model, N=4, batch_size=4, num_epochs=9999, earlyStoppingEpoches=40, device=0, date=None, encoding="None", sine_L=6, shuffle=True):

    img = readImg3D(img_dir)

    print("Preparing dataloader3D......")
    train_loader = dataloader3D.PixelLoader3D(img, N=N, batch_size=batch_size, encoding=encoding, sine_L=sine_L, shuffle=shuffle)
    print("Prepare finished!")
    model.to(device)

    loss_fn = nn.MSELoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # save dirs
    save_dir = f"./results/{date}/"
    os.makedirs(save_dir, exist_ok=True)
    save_dir_img = os.path.join(save_dir, "imgs")
    os.makedirs(save_dir_img, exist_ok=True)
    summary_dir = os.path.join(save_dir, "tensorboard")
    os.makedirs(summary_dir, exist_ok=True)
    model_dir = os.path.join(save_dir, "saved_models")
    os.makedirs(model_dir, exist_ok=True)

    print("Loading img_high and img_low......")
    img_high = train_loader.get_high_img()
    img_low = train_loader.get_low_img()
    saveImg3D(save_dir_img, "reference", img_high)
    saveImg3D(save_dir_img, "train", img_low)
    print("Loading finished!")

    print("start training...")
    MSE_history = list()
    PSNR_history = list()
    SSIM_history = list()
    loss_history = list()
    for epoch in range(num_epochs):
        # train
        model.train()
        loss_steps = []
        print(f"epoch{epoch}: ", end="")
        for i, (batch_x, batch_y) in enumerate(train_loader.load_train()):
            # train
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_steps.append(loss.item())
            print(f"\repoch {epoch}/{num_epochs}: batch {i}/{train_loader.batch_num}, loss:{np.mean(loss_steps)}", end="")
        torch.save(model, os.path.join(model_dir, f"model_at_epoch{epoch+1}.pt"))
        # # evaluate
        # model.eval()
        # print(f"\rfitting image...", end="")
        # img_fit = train_loader.fit_img(model)
        # # save img
        # print(f"\rsaving img...", end="")
        # saveImg3D(save_dir_img, f"{epoch}", img_fit)
        #
        # print(f"\rcalculating criterion...", end="")
        # MSE = criterion.mse(img_high, img_fit)
        # MSE_history.append(MSE)
        # PSNR = criterion.psnr(img_high, img_fit)
        # PSNR_history.append(PSNR)
        # SSIM = criterion.ssim(img_high, img_fit)
        # SSIM_history.append(SSIM)
        LOSS = np.mean(loss_steps)
        loss_history.append(LOSS)
        #
        # # save best model
        # if MSE_history[-1] <= np.min(MSE_history):
        #     torch.save(model, save_dir+"best_model.pt")
        #
        # # write summary
        # print(f"\rwriting summary to tensorboard...", end="")
        summaryWriter = SummaryWriter(summary_dir)
        summaryWriter.add_scalars("train loss", {"train loss": loss_history[-1]}, epoch)
        # summaryWriter.add_scalars("fit MSE", {"MSE": MSE_history[-1]}, epoch)
        # summaryWriter.add_scalars("fit PSNR", {"PSNR": PSNR_history[-1]}, epoch)
        # summaryWriter.add_scalars("fit SSIM", {"SSIM": SSIM_history[-1]}, epoch)
        #
        # print(f"\repoch {epoch}/{num_epochs}: loss={loss_history[-1]:.4f}, MSE={MSE:.4f}, PSNR={PSNR:.4f}, SSIM={SSIM:.4f}")
        #
        # if ( (epoch - MSE_history.index(min(MSE_history)) - 1) == earlyStoppingEpoches ):
        #     break


    # # write summary test
    # best_mse_index = MSE_history.index(min(MSE_history))
    # result = f"best MSE at epoch{best_mse_index+1}: mse={MSE_history[best_mse_index]}, psnr={PSNR_history[best_mse_index]}, ssim={SSIM_history[best_mse_index]}, loss={np.min(loss_history)}"
    # with open(os.path.join(save_dir, "result.txt"), "w") as file:
    #     file.write(result)
    #
    # # final fit
    # model = torch.load(save_dir+"best_model.pt")
    # model.eval()
    # img_fit = train_loader.fit_img(model)
    # saveImg3D(save_dir_img, "best", img_fit)
    #
    # MSE = criterion.mse(img_high, img_fit)
    # PSNR = criterion.psnr(img_high, img_fit)
    # SSIM = criterion.ssim(img_high, img_fit)
    #
    # print(f"final model saved, with MSE={MSE:.4f}, PSNR={PSNR:.4f}, SSIM={SSIM:.4f}")
    #
    # return MSE, PSNR, SSIM