'''
@Descripttion: This is Aoru Xue's demo,which is only for reference
@version: 
@Author: Aoru Xue
@Date: 2019-06-15 12:56:39
@LastEditors: Aoru Xue
@LastEditTime: 2019-09-10 20:46:54
'''
import torch
import torchvision
from TinySSD import TinySSD
#from vgg_ssd import build_ssd_model
from dataset import Mydataset
from torchvision import transforms
#from transforms import *
from torch.utils.data import DataLoader
from multibox_loss import MultiBoxLoss
import torch.optim as optim
from tqdm import tqdm
def train(dataloader,net,loss_fn,optimizer,epochs = 200):
    for epoch in range(epochs):
        running_loss_bbox = 0.
        running_loss_class = 0.
        for img,gt_bbox,gt_class in tqdm(dataloader):
            img = img.cuda()
            gt_bbox = gt_bbox.cuda()
            gt_class = gt_class.cuda()
            optimizer.zero_grad()
            pred_class,pred_locations = net(img)
            """Compute classification loss and smooth l1 loss.

                Args:
                    confidence (batch_size, num_priors, num_classes): class predictions.
                    predicted_locations (batch_size, num_priors, 4): predicted locations.
                    labels (batch_size, num_priors): real labels of all the priors.
                    gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
            """
            regression_loss, classification_loss = loss_fn(pred_class ,pred_locations,gt_class,gt_bbox)
            loss = regression_loss + classification_loss
            loss.backward()
            running_loss_bbox += regression_loss.item()
            running_loss_class += classification_loss.item()
            optimizer.step()
            #print(pred_bbox.size(),pred_class.size())
            
            #print("epoch: {},bbox loss:{:.8f} , class loss:{:.8f}".format(epoch + 1,loss[0].cpu().item(),loss[1].cpu().item()))
        print("*" * 20)
        print("average bbox loss: {:.8f}; average class loss: {:.8f}".format(running_loss_bbox/len(dataloader),running_loss_class/len(dataloader)))
        if epoch % 5 == 0:
            torch.save(net.state_dict(),"./ckpt/{}.pkl".format(epoch))
if __name__ == "__main__":
    net = TinySSD()
    net.cuda()
    loss_fn = MultiBoxLoss(3.)
    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    )
    # transform = Compose([
    #     ConvertFromInts(),
    #     PhotometricDistort(),
    #     Expand([123, 117, 104]),
    #     RandomSampleCrop(),
    #     RandomMirror(),
    #     ToPercentCoords(),
    #     Resize(300),
    #     SubtractMeans([123, 117, 104]),
    #     ToTensor(),
    # ])

    optm = optim.Adam(net.parameters(),lr = 1e-3)
    dtset = Mydataset(img_path = "./dataset",transform = transform)
    dataloader = DataLoader(dtset,batch_size = 8,shuffle = True)
    
    train(dataloader,net,loss_fn,optm)
