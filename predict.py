'''
@Descripttion: This is Aoru Xue's demo,which is only for reference
@version: 
@Author: Aoru Xue
@Date: 2019-09-02 21:08:56
@LastEditors: Aoru Xue
@LastEditTime: 2019-09-04 19:22:18
'''
import torch
import torchvision
#from vgg_ssd import build_ssd_model
from TinySSD import TinySSD
from torchvision import transforms
#from transforms import *
from PIL import Image
from viz import draw_bounding_boxes
from post_processor import PostProcessor
post_process = PostProcessor()
transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    )
# predict_transform = Compose([
#             Resize(300),
#             SubtractMeans([123, 117, 104]),
#             ToTensor()
#         ])
import numpy as np
def center_form_to_corner_form(locations):
    return np.concatenate([locations[..., :2] - locations[..., 2:] / 2,
                      locations[..., :2] + locations[..., 2:] / 2], 1)
if __name__ == "__main__":
    img = Image.open("./dataset/images/045346.jpg").convert('RGB')
    image = np.array(img,dtype = np.float32)
    height, width, _ = image.shape
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.cuda()
    net = TinySSD()
    
    
    net.load_state_dict(torch.load("./ckpt/170.pkl"))
    net.cuda()
    net.eval()
    with torch.no_grad():
        pred_confidence,pred_bbox = net(img)
        #print(pred_confidence)
        pred_bbox = center_form_to_corner_form(pred_bbox)
        output = post_process(pred_confidence,pred_bbox, width=width, height=height)[0]
        
        boxes, labels, scores = [o.to("cpu").numpy() for o in output]
        print(scores)
        
        drawn_image = draw_bounding_boxes(image, boxes, labels, scores, ("__background__","basketball","volleyball")).astype(np.uint8)
        
        Image.fromarray(drawn_image).save("./a.jpg")
