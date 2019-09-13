'''
@Descripttion: This is Aoru Xue's demo,which is only for reference
@version: 
@Author: Aoru Xue
@Date: 2019-09-02 23:21:38
@LastEditors: Aoru Xue
@LastEditTime: 2019-09-10 20:11:39
'''
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
from multibox_loss import MultiBoxLoss
from prior_box import PriorBox
import box_utils
class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class SSD(nn.Module):
    def __init__(self,
                 vgg: nn.ModuleList,
                 extras: nn.ModuleList,
                 classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()
        self.num_classes = 3
        self.vgg = vgg
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.l2_norm = L2Norm(512, scale=20)
        self.priors = None
    def reset_parameters(self):
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.vgg.apply(weights_init)
        self.extras.apply(weights_init)
        self.classification_headers.apply(weights_init)
        self.regression_headers.apply(weights_init)

    def forward(self, x, targets=None):
        sources = []
        confidences = []
        locations = []
        for i in range(23):
            x = self.vgg[i](x)
        s = self.l2_norm(x)  # Conv4_3 L2 normalization
        sources.append(s)

        # apply vgg up to fc7
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
        sources.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        for (x, l, c) in zip(sources, self.regression_headers, self.classification_headers):
            locations.append(l(x).permute(0, 2, 3, 1).contiguous())
            confidences.append(c(x).permute(0, 2, 3, 1).contiguous())

        confidences = torch.cat([o.view(o.size(0), -1) for o in confidences], 1)
        locations = torch.cat([o.view(o.size(0), -1) for o in locations], 1)

        confidences = confidences.view(confidences.size(0), -1, self.num_classes)
        locations = locations.view(locations.size(0), -1, 4)

        if not self.training:
            # when evaluating, decode predictions
            if self.priors is None:
                self.priors = PriorBox()().to(locations.device)
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, 0.1, 0.2
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            print("testing !")
            return confidences, boxes
        else:
            return (confidences, locations)

    def init_from_base_net(self, model):
        vgg_weights = torch.load(model, map_location=lambda storage, loc: storage)
        self.vgg.load_state_dict(vgg_weights, strict=True)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


