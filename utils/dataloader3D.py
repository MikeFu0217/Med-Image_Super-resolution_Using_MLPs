import numpy as np
import torch
import torch.nn as nn
import math
import random
from itertools import product


class PixelLoader3D:
    def __init__(self, img_high, N, batch_size, shuffle=True, device=0, encoding="None", sine_L=6):
        # image of high resolution (original image)
        self.img_high = img_high/255.
        self.N = N
        self.org_height = self.img_high.shape[0]
        self.org_width = self.img_high.shape[1]
        self.org_length = self.img_high.shape[2]
        self.norm = max(max(self.org_height-1, self.org_width-1), self.org_length-1)
        self.data_high = [(i[0], i[1], i[2], self.img_high[i[0], i[1], i[2]]) for i in product(range(self.org_height), range(self.org_width), range(self.org_length))]
        # image of low resolution
        self.low_height = math.ceil(self.img_high.shape[0]/self.N)
        self.low_width = math.ceil(self.img_high.shape[1]/self.N)
        self.low_length = math.ceil(self.img_high.shape[2]/self.N)
        self.data_low = []
        for i in range(self.img_high.shape[0]):
            for j in range(self.img_high.shape[1]):
                for k in range(self.img_high.shape[2]):
                    if i % N == 0 and j % N == 0 and k % N == 0:
                        self.data_low.append((i, j, k, self.img_high[i][j][k]))
        self.batch_size = batch_size
        self.batch_num = len(self.data_low) // self.batch_size
        self.shuffle = shuffle
        self.device = device
        self.encoding = encoding
        self.sine_L = sine_L

    def get_high_img(self):
        # img = np.zeros((self.org_height, self.org_width, self.org_length))
        # for i in self.data_high:
        #     img[i[0], i[1], i[2]] = round(i[3]*255)
        # return img
        img = np.array([i[3] for i in self.data_high]).reshape((self.org_height, self.org_width, self.org_length), order='A')
        return (img*255).astype(int)


    def get_low_img(self):
        # img = np.zeros((self.low_height, self.low_width, self.low_length))
        # for i in self.data_low:
        #     img[i[0] // self.N, i[1] // self.N, i[2] // self.N] = round(i[3]*255)
        # return img
        img = np.array([i[3] for i in self.data_low]).reshape((self.low_height, self.low_width, self.low_length), order='A')
        return (img*255).astype(int)


    def fit_img(self, model):
        img = np.array([])
        pixels_all = list(product(range(self.org_height), range(self.org_width), range(self.org_length)))
        batchsize = self.org_width*self.org_height*self.org_length//10

        for iter in range(len(pixels_all)//batchsize):
            pixels = pixels_all[iter*batchsize:(iter+1)*batchsize]
            input = [self.positionEncode(i[0], i[1], i[2], self.encoding) for i in pixels]
            input = torch.tensor(np.array(input).astype(np.float32)).to(self.device)
            output = model(input).detach().cpu().numpy()
            img = output if iter==0 else np.concatenate((img, output), axis=0)
            print(f"\rfitting image {iter}/{len(pixels_all)//batchsize}", end="")
        pixels = pixels_all[(iter+1)*batchsize:]

        if (len(pixels)!=0):
            input = [self.positionEncode(i[0], i[1], i[2], self.encoding) for i in pixels]
            input = torch.tensor(np.array(input).astype(np.float32)).to(self.device)
            output = model(input).detach().cpu().numpy()
            img = np.concatenate((img, output), axis=0)
            print(f"\rfitting image {iter+1}/{len(pixels_all) // batchsize}", end="")

        img = img.reshape((self.org_height, self.org_width, self.org_length))

        return (img*255).astype(int)

    def positionEncode(self, i, j, k, mode):
        # normalize to [0,1]
        ## x = (j - (self.org_width-1)/2) / self.norm * 2
        ## y = (i - (self.org_height-1)/2) / self.norm * 2
        x = j / self.norm
        y = i / self.norm
        z = k / self.norm
        # encode
        if mode == "None":
            return np.array([x, y, z])
        elif mode == "Sine":
            L = self.sine_L
            encodeX = []
            for i in range(L):
                encodeX.append(np.sin(2**i*np.pi*x))
                encodeX.append(np.cos(2**i*np.pi*x))
            encodeY = []
            for i in range(L):
                encodeY.append(np.sin(2**i*np.pi*y))
                encodeY.append(np.cos(2**i*np.pi*y))
            encodeZ = []
            for i in range(L):
                encodeZ.append(np.sin(2 ** i * np.pi * z))
                encodeZ.append(np.cos(2 ** i * np.pi * z))
            result = np.array([encodeX, encodeY, encodeZ])
            # #print(f"\rencode with ({x},{y},{z})")
            # position = np.array([x, y, z])
            # encodings = []
            # for i in range(L):
            #     encodings.append(np.sin(2 ** i * np.pi * position))
            #     encodings.append(np.cos(2 ** i * np.pi * position))
            # result = np.array([encodings])
            return result
        elif mode == "tranf":
            D = 20
            positions = np.array([x,y,z])
            encodings = []
            for i in range(D // 2):
                for pos in positions:
                    encodings.append(np.sin(pos / (self.norm ** (2 * i / D))))
                    encodings.append(np.cos(pos / (self.norm ** (2 * i / D))))
            result = np.array([encodings])
            return result

    def load_train(self):
        if self.shuffle:
            random.shuffle(self.data_low)
        for i in range(self.batch_num):
            batch_data = self.data_low[i * self.batch_size:(i + 1) * self.batch_size]
            batch_x = [self.positionEncode(a[0], a[1], a[2], self.encoding) for a in batch_data]
            batch_x = torch.tensor(np.array(batch_x).astype(np.float32)).to(self.device)
            batch_y = [a[3] for a in batch_data]
            batch_y = torch.tensor(np.array(batch_y).astype(np.float32)).to(self.device).unsqueeze(-1)
            yield batch_x, batch_y
