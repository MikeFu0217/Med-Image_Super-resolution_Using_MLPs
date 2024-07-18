import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils import dataloader, criterion
import os
from torch.utils.tensorboard import SummaryWriter

def train_and_evaluate(img_dir, model, N=4, batch_size=4, num_epochs=9999, earlyStoppingEpoches=40, device=0, date=None, encoding="None", sine_L=6, shuffle=True):
    img = plt.imread(img_dir)
    img = img if img.shape[-1]!=3 else img.mean(axis=2)
    train_loader = dataloader.PixelLoader(img, N=N, batch_size=batch_size, encoding=encoding, sine_L=sine_L, shuffle=shuffle)
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

    img_high = train_loader.get_high_img()
    img_low = train_loader.get_low_img()
    plt.imsave(os.path.join(save_dir_img, f"reference.jpg"), img_high, cmap="gray")
    plt.imsave(os.path.join(save_dir_img, f"train.jpg"), img_low, cmap="gray")

    print("start training...")
    MAE_history = list()
    PSNR_history = list()
    SSIM_history = list()
    loss_history = list()
    for epoch in range(1, num_epochs + 1):
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

        # evaluate
        model.eval()
        img_fit = train_loader.fit_img(model)

        MAE = criterion.mae(img_high, img_fit)
        MAE_history.append(MAE)
        PSNR = criterion.psnr(img_high, img_fit)
        PSNR_history.append(PSNR)
        SSIM = criterion.ssim(img_high, img_fit, multichannel=False)
        SSIM_history.append(SSIM)
        LOSS = np.mean(loss_steps)
        loss_history.append(LOSS)

        # save img
        plt.imsave(os.path.join(save_dir_img, f"{epoch}.jpg"), img_fit, cmap="gray")
        # save best model
        if PSNR_history[-1] > np.max(PSNR_history):
            torch.save(model, save_dir+"best_model.pt")

        # write summary
        summaryWriter = SummaryWriter(summary_dir)
        summaryWriter.add_scalars("train loss", {"train loss": loss_history[-1]}, epoch)
        summaryWriter.add_scalars("fit MAE", {"MAE": MAE_history[-1]}, epoch)
        summaryWriter.add_scalars("fit PSNR", {"PSNR": PSNR_history[-1]}, epoch)
        summaryWriter.add_scalars("fit SSIM", {"SSIM": SSIM_history[-1]}, epoch)

        print(f"\repoch {epoch}/{num_epochs}: loss={loss_history[-1]:.4f}, MAE={MAE:.4f}, PSNR={PSNR:.4f}, SSIM={SSIM:.4f}")

        if ( (epoch - PSNR_history.index(max(PSNR_history)) - 1) == earlyStoppingEpoches ):
            break


    # write summary test
    best_psnr_index = PSNR_history.index(max(PSNR_history))
    result = f"best PSNR at epoch{best_psnr_index+1}: mse={MAE_history[best_psnr_index]}, psnr={PSNR_history[best_psnr_index]}, ssim={SSIM_history[best_psnr_index]}, loss={loss_history[best_psnr_index]}"
    with open(os.path.join(save_dir, "result.txt"), "w") as file:
        file.write(result)

    # final fit
    model = torch.load(save_dir+"best_model.pt")
    model.eval()
    img_fit = train_loader.fit_img(model)
    plt.imsave(os.path.join(save_dir_img, f"best.jpg"), img_fit, cmap="gray")

    MAE = criterion.mae(img_high, img_fit)
    PSNR = criterion.psnr(img_high, img_fit)
    SSIM = criterion.ssim(img_high, img_fit, multichannel=True)

    print(f"final model saved, with MAE={MAE:.4f}, PSNR={PSNR:.4f}, SSIM={SSIM:.4f}")

    return MAE, PSNR, SSIM