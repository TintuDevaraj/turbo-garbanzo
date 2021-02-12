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


parser = argparse.ArgumentParser()
parser.add_argument('--arch',dest="arch",action="store",default="vgg16",type = str)
parser.add_argument('data_dir',type=str,default='flowers/',help ='directory of the files')
parser.add_argument('--save_dir',dest="save_dir",action="store",default='./checkpoint.pth',help="directory to save model")
parser.add_argument('--learning_rate',dest="learning_rate",action="store",default=0.001,help="learning rate of the model")
parser.add_argument('--hidden_units',type = int,dest="hidden_units",action="store",default=[1024,512],help="hidden units of the model")
parser.add_argument('--epochs',dest="epochs",action="store",type = int,default=4,help="epochs to train the model")
parser.add_argument('--gpu',dest="gpu",action="store",default=False)
args= parser.parse_args()

def dataloader(data_dir):
    data_transforms = {
        'train':transforms.Compose([transforms.RandomResizedCrop(size=224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                               ]),
        'valid':transforms.Compose([transforms.Resize(size=256),
                                transforms.CenterCrop(size=224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                               ]),
        'test':transforms.Compose([transforms.Resize(size=256),
                                transforms.CenterCrop(size=224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                               ]),
    }

    image_datasets = {'train': datasets.ImageFolder(train_dir,transform = data_transforms['train']),
                  'valid': datasets.ImageFolder(valid_dir,transform = data_transforms['valid']),
                  'test': datasets.ImageFolder(test_dir,transform = data_transforms['test']),
                 }
    data_loader = {train_loader: torch.utils.data.DataLoader(image_datasets['train'],batch_size = 64,shuffle=True),
                   validation_loader: torch.utils.data.DataLoader(image_datasets['valid'],batch_size = 64,shuffle=True),
                   test_loader:torch.utils.data.DataLoader(image_datasets['test'],batch_size = 64,shuffle=True)
                  }  
    data_size = {train_loader_size:len(data_loader[train_loader]),
                 validation_loader_size:len(data_loader[validation_loader]),
                 test_loader_size:len(data_loader[test_loader])
                }
    return data_loader,data_size


def model(hidden_units,learning_rate,arch=,device="cpu"):
    
    t_models={'vgg19':models.vgg19(pretrained=True),
              'densenet121':models.densenet121(pretrained=True),
              'resnet101':models.resnet101(pretrained=True)
             }
    model = t_models.get(model_name,'vgg19')
    classifier=None
    optimizer=None
    
    for param in model.parameters():
        param.requires_grad=False
    
        from collections import OrderedDict
        if model_name=='vgg19':
            classifier=nn.Sequential(OrderedDict([
                ('fc1',nn.Linear(25088,4096)),
                ('relu1',nn.ReLU()),
                ('dropout1',nn.Dropout(p=0.4)),
                ('fc2',nn.Linear(4096,1024)),
                ('relu2',nn.ReLU()),
                ('dropout2',nn.Dropout(p=0.4),
                ('fc4',nn.Linear(1024,102)),
                ('output',nn.LogSoftmax(dim=1))
                ]))
            model.classifier = classifier
            optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)    
        elif model_name=='densenet121':
            classifier=nn.Sequential(OrderedDict([
                ('fc1',nn.Linear(1024,1000)),
                ('relu1',nn.ReLU()),
                ('dropout1',nn.Dropout(p=0.4)),
                ('fc2',nn.Linear(1000,512)),
                ('relu2',nn.ReLU()),
                ('dropout2',nn.Dropout(p=0.4),
                ('fc4',nn.Linear(512,102)),
                ('output',nn.LogSoftmax(dim=1))
                ]))
            model.classifier = classifier
            optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)
        elif model_name == 'resnet101':
            classifier=nn.Sequential(OrderedDict([
                ('fc1',nn.Linear(2048,1000)),
                ('relu1',nn.ReLU()),
                ('dropout1',nn.Dropout(p=0.4)),
                ('fc2',nn.Linear(1000,512)),
                ('relu2',nn.ReLU()),
                ('dropout2',nn.Dropout(p=0.4),
                ('fc4',nn.Linear(512,102)),
                ('output',nn.LogSoftmax(dim=1))
                ]))
            model.classifier = classifier
            optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)
    
    criterion = nn.NLLLoss()
    model.to(device)
        
    return(model,criterion,optimizer)
    
def main(): 
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader,data_size = dataloader(args.data_dir)
    model,criterion,optimizer = model(hidden_units = args.hidden_units,learning_rate = args.learning_rate,arch=args.arch,device)
        
        
    epochs=args.epochs


    for epoch in range (epochs):
        print("Epoch: {}/{}".format(epoch+1,epochs))
        model.train()
        running_loss=0.0
    
        
        for i, (images,labels) in enumerate(train_loader):
            images,labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
            
            logps=model(images)
            loss=criterion(logps,labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item() * images.size(0) 
            print(" Train Loss: {:.4f}".format(loss.item()))
        
        
            valid_loss = 0.0
            valid_acc = 0.0
            with torch.no_grad():
                model.eval()
                
                
                for ii,(images,labels) in enumerate(validation_loader):
                    optimizer.zero_grad()
                    images,labels = images.to(device),labels.to(device)
                    logps = model.forward(images)
                    loss = criterion(logps,labels)
                    valid_loss += loss.item() * images.size(0)
                
                    ret,predictions = torch.max(logps.data,1)
                    equality=predictions.eq(labels.data.view_as(predictions))
                    acc = torch.mean(equality.type(torch.FloatTensor))
                    valid_acc += acc.item() * images.size(0)
                    print(" validation Loss:{:.4f}".format(loss.item()),
                        " accuracy: {:.4f}".format(acc.item()))
                
    def create_checkpoint(model,path,model_name,class_to_idx,optimizer,epochs):
        model.cpu()
        model.class_to_idx = class_to_idx
        checkpoint = {'arch' :model_name,
                'class_to_idx':model.class_to_idx,
                'running_loss':running_loss,
                'valid_loss':valid_loss,
                'dropout' :'0.5',
                'epochs' : epochs,
                'state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict()}
        
        if model_name == 'resnet101':
            checkpoint['fc'] = model.fc
        else:
            checkpoint['classifier'] = model.classifier
        file= str()
        if path != None:
            file= path+ '/' + model_name + '_checkpoint.pth'
        else:
            file= model_name + '_checkpoint.pth'            
        torch.save(checkpoint,file)
    
if _name_ == '_main_':
    main()
