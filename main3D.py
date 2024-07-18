import os.path
import sys
import datetime
from models import NN, resnet, Siren
from utils import train_tools3D
import pandas as pd


date_org = str(datetime.date.today())+"-"+str(datetime.datetime.now().hour)

if __name__ == '__main__':
    # data_folder = sys.argv[1]
    # img_dir = os.path.join(data_folder, '0.bmp')
    img_dir = "./data/Carotid/high"

    grid_search = False
    if not grid_search:
        L = 6
        model = NN.MLP(6 * L, 256)
        # model = resnet.ResNet1D()
        # model = Siren.Siren(in_features=3, out_features=1, hidden_features=256, hidden_layers=3, outermost_linear=True, first_omega_0=50., hidden_omega_0=50.)
        train_tools3D.train_and_evaluate(img_dir, model, N=1, batch_size=512, num_epochs=400, earlyStoppingEpoches=40, device=0, date=date_org, encoding="Sine", sine_L=L)
    else:
        Ls = [4, 5, 6, 7, 8, 9, 10, 11, 12]
        Sizes = [64, 128, 256, 512, 1024, 2048]
        GRID_MSE = pd.DataFrame(index=Ls, columns=Sizes)
        GRID_PSNR = pd.DataFrame(index=Ls, columns=Sizes)
        GRID_SSIM = pd.DataFrame(index=Ls, columns=Sizes)
        for L in Ls:
            for Size in Sizes:
                print("L={}, Size={}".format(L, Size))
                date = date_org + f"-L{L}-Size{Size}"
                model = NN.MLP(4*L, Size)
                mse, psnr, ssim = train_tools3D.train_and_evaluate3D(img_dir, model, N=4, batch_size=8, num_epochs=9999, earlyStoppingEpoches=40, device=0, date=date, encoding="Sine", sine_L=L)
                GRID_MSE[Size][L] = mse
                GRID_PSNR[Size][L] = psnr
                GRID_SSIM[Size][L] = ssim
                GRID_MSE.to_csv("GRID_MSE.csv")
                GRID_PSNR.to_csv("GRID_PSNR.csv")
                GRID_SSIM.to_csv("GRID_SSIM.csv")