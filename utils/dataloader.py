import numpy as np
import torch
import torch.nn as nn
import math
import random
from itertools import product


class PixelLoader:
    def __init__(self, img_high, N, batch_size, shuffle=True, device=0, encoding="None", sine_L=6):
        # image of high resolution (original image)
        self.img_high = img_high/255.
        self.N = N
        self.org_height = self.img_high.shape[0]
        self.org_width = self.img_high.shape[1]
        self.norm = self.org_height-1 if self.org_height>self.org_width else self.org_width-1
        self.data_high = [(i[0], i[1], self.img_high[i[0], i[1]]) for i in product(range(self.org_height), range(self.org_width))]
        # image of low resolution
        self.data_low = []
        for i in range(self.img_high.shape[0]):
            for j in range(self.img_high.shape[1]):
                if i % N == 0 and j % N == 0:
                    self.data_low.append((i, j, self.img_high[i][j]))
        self.batch_size = batch_size
        self.batch_num = len(self.data_low) // self.batch_size
        self.shuffle = shuffle
        self.device = device
        self.encoding = encoding
        self.sine_L = sine_L

    def get_high_img(self):
        img = np.zeros((self.img_high.shape[0], self.img_high.shape[1]))
        for i in self.data_high:
            img[i[0], i[1]] = round(i[2]*255)
        return img

    def get_low_img(self):
        img = np.zeros((math.ceil(self.img_high.shape[0]/self.N), math.ceil(self.img_high.shape[1]/self.N)))
        for i in self.data_low:
            img[i[0] // self.N, i[1] // self.N] = round(i[2]*255)
        return img

    def fit_img(self, model):
        img = np.zeros((self.org_height, self.org_width))
        pixels = list(product(range(self.org_height), range(self.org_width)))
        input = [self.positionEncode(i[0], i[1], self.encoding) for i in pixels]
        input = torch.tensor(np.array(input).astype(np.float32)).to(self.device)
        output = model(input).detach().cpu().numpy()
        for id in range(len(pixels)):
            i = pixels[id][0]
            j = pixels[id][1]
            value = output[id][0]
            img[i, j] = round(value*255)
        return img

    def positionEncode(self, i, j, mode):
        # normalize to [0,1]
        ## x = (j - (self.org_width-1)/2) / self.norm * 2
        ## y = (i - (self.org_height-1)/2) / self.norm * 2
        x = j / self.norm
        y = i / self.norm
        # encode
        if mode == "None":
            return np.array([x, y])
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
            result = np.array([encodeX, encodeY])
            return result
        elif mode == "tranf":
            D = 20
            positions = np.array([x,y])
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
            batch_x = [self.positionEncode(a[0], a[1], self.encoding) for a in batch_data]
            batch_x = torch.tensor(np.array(batch_x).astype(np.float32)).to(self.device)
            batch_y = [a[2] for a in batch_data]
            batch_y = torch.tensor(np.array(batch_y).astype(np.float32)).to(self.device).unsqueeze(-1)
            yield batch_x, batch_y
