'''
@Descripttion: This is Aoru Xue's demo,which is only for reference
@version: 
@Author: Aoru Xue
@Date: 2019-06-14 00:42:10
@LastEditors: Aoru Xue
@LastEditTime: 2019-09-02 17:04:26
'''


import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from prior_box import PriorBox
from torchsummary import summary
import torch.nn.functional as F
from box_utils import *
from PIL import Image
class TinySSD(nn.Module):
    def __init__(self,training = True):
        super(TinySSD,self).__init__()
        self.basenet = EfficientNet.from_name('efficientnet-b0')
        self.training = training
        for idx,num_anchors in enumerate([4, 6, 6, 4, 4]):
            setattr(self,"predict_bbox_{}".format(idx + 1),nn.Conv2d(
                320,num_anchors * 4,kernel_size = 3,padding = 1
            ))
            setattr(self,"predict_class_{}".format(idx + 1),nn.Conv2d( # 这里3 是 2 + 1
                320,3 * num_anchors,kernel_size = 3,padding = 1
            ))
        self.priors = None
        for idx,k in enumerate([[320,320],[320,320],[320,320]]):
            setattr(self,"feature_{}".format(idx + 2),nn.Sequential(
                nn.Conv2d(k[0],k[1],kernel_size = 3,padding =1),
                nn.BatchNorm2d(k[1]),
                nn.ReLU(),
                nn.Conv2d(k[1],k[1],kernel_size = 3,padding =1),
                nn.BatchNorm2d(k[1]),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ))
    def forward(self,x):
        x = self.basenet.extract_features(x)
        feature_1 = x
        feature_2 = self.feature_2(x)
        feature_3 = self.feature_3(feature_2)
        feature_4 = self.feature_4(feature_3)
        feature_5 = F.max_pool2d(feature_4,kernel_size = 2)
        
        
        '''
        (2,4*4,16,16)
        (2,4*6,8,8)
        (2,4*6,4,4),
        (2,4*4,2,2),
        (2,4*4,1,1)

        -> 每个 anchor 中心，连续4个值代表x y w h
        '''
        confidences = []
        locations = []
        locations.append(self.predict_bbox_1(feature_1).permute(0,2,3,1).contiguous())
        locations.append(self.predict_bbox_2(feature_2).permute(0,2,3,1).contiguous())
        locations.append(self.predict_bbox_3(feature_3).permute(0,2,3,1).contiguous())
        locations.append(self.predict_bbox_4(feature_4).permute(0,2,3,1).contiguous())
        locations.append(self.predict_bbox_5(feature_5).permute(0,2,3,1).contiguous())
        locations = torch.cat([o.view(o.size(0), -1) for o in locations], 1) #(batch_size,total_anchor_num*4)
        locations = locations.view(locations.size(0), -1, 4) # (batch_size,total_anchor_num,4)

        confidences.append(self.predict_class_1(feature_1).permute(0,2,3,1).contiguous())
        confidences.append(self.predict_class_2(feature_2).permute(0,2,3,1).contiguous())
        confidences.append(self.predict_class_3(feature_3).permute(0,2,3,1).contiguous())
        confidences.append(self.predict_class_4(feature_4).permute(0,2,3,1).contiguous())
        confidences.append(self.predict_class_5(feature_5).permute(0,2,3,1).contiguous())
        confidences = torch.cat([o.view(o.size(0), -1) for o in confidences], 1) #(batch_size,total_anchor_num*4)
        confidences = confidences.view(confidences.size(0), -1, 3) # (batch_size,total_anchor_num,4)
        if not self.training:
            if self.priors is None:
                self.priors = PriorBox()()
                self.priors = self.priors.cuda()
            boxes = convert_locations_to_boxes(
                locations, self.priors, 0.1, 0.2
            )
            confidences = F.softmax(confidences, dim=2)
            return confidences, boxes
        else:
            #print(confidences.size(),locations.size())
            return (confidences, locations) #  (2,1111,3) (2,1111,4)
        
if __name__ == "__main__":
    net = TinySSD()
    net.cuda()
    #prior = PriorBox()
    #print(len(prior()))
    #gt_prior = assign_priors(torch.Tensor([[0,0,10/512,10/512],[55/512,55/512,30/512,30/512]]),torch.Tensor([1,2,5]),prior(),0.5)
    #print(gt_prior[1])
    #x = torch.randn(1,3,512,512)
    #out = net(x.cuda())
    #print(out[0].size())
    #print(out[1].size())
    #print(prior()[:200,:])
    #print(out[0][0])
    #print(out[1][0])
    summary(net,(3,512,512),device="cuda")

    

