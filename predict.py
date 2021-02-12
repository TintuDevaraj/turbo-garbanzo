import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch import tensor
from torch import optim
import torchvision.datasets as datasets,
import torchvision.transforms as transforms
import torchvision.models as models
import PIL
from PIL import Image
from collections import OrderedDict
import argparse
import torch.nn.functional as F
from torch.autograd import Variable
import json
from train import train


parser = argparse.ArgumentParser()
parser.add_argument('--image',type = str,help='Point  to image file for prediction',required = True)
parser.add_argument('--checkpoint',type = str,help='Point  to checkpoint file as str',required = True)
parser.add_argument('--top_k',dest='topk',action='store',default=5,help='Choose top k matches as int')
parser.add_argument('--category_names',dest = "category_names",action = "store",default = 'cat_to_name.json')
parser.add_argument('--gpu',default=False,action="store",dest="gpu")
    
args = parser.parse_args()



def load_checkpoint(path):
    
    t_models={'vgg19':models.vgg19(pretrained=True),
              'densenet121':models.densenet121(pretrained=True),
              'resnet101':models.resnet101(pretrained=True)
             }
    model = t_models.get(model_name,'vgg19')
    for param in model.parameters():
        param.requires_grad = False
    if checkpoint['arch'] == 'vgg19' or checkpoint['arch'] == densenet121:
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
    else:
        model.fc = checkpoint['fc']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def processimage(image):
    pil_image = Image.open(image)
    
    process_image = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485,0.456,0.406],
                                                             std=[0.229,0.224,0.225])
                                       ])
    np_image = process_image(pil_image)
    
    return np_image

def main():
    
    model = load_checkpoint('model_checkpoint.pth')
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
   
    model.to(device)
    image = process_image(image).to(device)
    
    np_image = image.unsqueeze_(0)
    model.eval()

    with torch.no_grad():
       logps = model.forward(np_image)
        ps = torch.exp(logps)
        top_k,top_classes_idx = ps.topk(topk,dim=1)
        top_k,top_classes_idx = np.array(top_k.to('cpu')[0]), np.array(top_classes_idx.to('cpu')[0])
    
        idx_to_class = {x:y for y,x in model.class_to_idx.items()}
    
        top_classes = []
        for index in top_classes_idx:
            top_classes.append(idx_to_class[index])
        
    

    if args.category_names != None:
        with open(args.category_names,'r')as f:
            cat_to_name = json.load(f)
            top_class_names = [cat_to_name[top] for top in list(top_classes)]
            print(probabilities:{list(top_k)},classes:{ list(top_class_names)})
        
    else:
        print(probabilities:{list(top_k)},classes:{ list(top_classes)})
        
        
if _name_ == '_main_':
    main()
    