import glob
import xml.etree.ElementTree as ET
class Datas():
    def __init__(self,img_path = "./dataset/train/images"):
        self.img_paths = glob.glob(img_path + "/*.jpg")
        self.labels = {
            "blue":1,
            "red":2
        }
    def label2int(self,label):
        return self.labels[label]
    def xml2txt(self,xml_file):
        file = ET.parse(xml_file)
        root = file.getroot()
        objs = root.findall("object")
        width,height = [int(root.find("size")[i].text) for i in range(2)]
        dw,dh = 1./width,1./height
        file = ""
        for obj in objs:
            xmin,ymin,xmax,ymax = [float(obj[4][i].text) for i in range(4)]
            label = obj[0].text
            #w,h = max(xmax-xmin,0),max(ymax-ymin,0)
            xmin,ymin,xmax,ymax = xmin*dw,ymin*dh,xmax*dw,ymax*dh
            file = file + "{} {} {} {} {}\n".format(xmin,ymin,xmax,ymax,self.label2int(label))
        return file
    def pose(self):
        for img in self.img_paths:
            xml_file = img.replace("images","annotations").replace(".jpg",".xml")
            txt = self.xml2txt(xml_file)
            with open(img.replace(".jpg",".txt"),"w") as f:
                f.write(txt)

if __name__ == "__main__":
    #imgs = glob.glob("./dataset/train/images/*.jpg")
    #with open("./dataset/train.txt","w") as f:
    #    for img in imgs:
    #        f.write(img + "\n")
    datas = Datas()
    datas.pose()
