'''
@Descripttion: This is Aoru Xue's demo,which is only for reference
@version: 
@Author: Aoru Xue
@Date: 2019-09-02 00:02:47
@LastEditors: Aoru Xue
@LastEditTime: 2019-09-10 19:47:33
'''
from itertools import product

import torch
import torch.nn as nn
from math import sqrt

class PriorBox(nn.Module):
    def __init__(self):
        super(PriorBox, self).__init__()
        self.image_size = 512
        self.feature_maps = [16,8,4,2,1]
        self.min_sizes = [30, 60, 111, 162, 213]
        self.max_sizes = [60, 111, 162, 213, 264]
        self.strides = [32, 64, 128,256, 512]
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2], [2]]
        self.clip = True

    def forward(self):
        """Generate SSD Prior Boxes.
            It returns the center, height and width of the priors. The values are relative to the image size
            Returns:
                priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        priors = []
        for k, f in enumerate(self.feature_maps): # every size of feature map
            scale = self.image_size / self.strides[k] # how many boxes (not anchor) in a row in raw img
            # 512 / 32 = 16
            for i, j in product(range(f), repeat=2): # xy generator in feature map
                # unit center x,y
                cx = (j + 0.5) / scale # see as blocks and xy in center of it 
                cy = (i + 0.5) / scale # 15,15 -> 15.5,15.5 -> 15.5/16,15.5/16 which means the xy in center of feature map

                # small sized square box
                size = self.min_sizes[k] # min size
                h = w = size / self.image_size # small size
                priors.append([cx, cy, w, h]) # the small size one

                # big sized square box
                size = sqrt(self.min_sizes[k] * self.max_sizes[k]) # the same as small one
                h = w = size / self.image_size
                priors.append([cx, cy, w, h])

                # change h/w ratio of the small sized box
                # considering the w/ratio ,  w*ratio , h/ratio and h * ratio
                size = self.min_sizes[k]
                h = w = size / self.image_size
                for ratio in self.aspect_ratios[k]:
                    ratio = sqrt(ratio)
                    priors.append([cx, cy, w * ratio, h / ratio])
                    priors.append([cx, cy, w / ratio, h * ratio])

        priors = torch.Tensor(priors)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors

if __name__ == "__main__":
    import random
    import numpy as np
    import cv2 as cv
    prior = PriorBox()
    img = np.zeros((300,300,3),dtype = np.uint8)
    priors = prior()
    print(priors.size())
    for x,y,w,h in priors:
        x = x * 300
        y = y * 300
        w = w*300
        h = h * 300
        if w < 100 and h < 100:
            cv.rectangle(img,(int(x.item() - w.item()//2),int(y.item() - h.item()//2)),(int(x.item() + w.item()//2),int(y.item() + h.item()//2)),(random.randint(0,255),random.randint(0,255),random.randint(0,255)),1)
    cv.imshow("img",img)
    cv.waitKey(0)
