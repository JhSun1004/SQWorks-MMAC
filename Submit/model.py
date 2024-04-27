import os
import cv2
import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms as T 
from PIL import Image
import numpy as np
from multi_model import EfficientNet

class model:
    def __init__(self):
        self.models = []
        self.checkpoint_names = []
        self.checkpoints = [
            ('efficientnetb0',"efficientb0.pkl"),
            ('efficientnetb1',"efficientb1.pkl"),
            ('efficientnetb2',"efficientb2.pkl"),
            ('efficientnetb3',"efficientb3.pkl"),
            ('resnet18',"resnet18.pkl"),
            ('resnet50',"resnet50.pkl")
            

            ]
        # The model is evaluated using CPU, please do not change to GPU to avoid error reporting.
        self.device = torch.device("cpu")
        # self.tta = 3

    def load(self, dir_path):
        """
        load the model and weights.
        dir_path is a string for internal use only - do not remove it.
        all other paths should only contain the file name, these paths must be
        concatenated with dir_path, for example: os.path.join(dir_path, filename).
        :param dir_path: path to the submission directory (for internal use only).
        :return:
        """
        for model_name, checkpoint_name in self.checkpoints:
            net = get_model(model_name)

            # join paths
            checkpoint_path = os.path.join(dir_path, checkpoint_name)
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt['net']
            unParalled_state_dict = {}
            for key in state_dict.keys():
                unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
            net.load_state_dict(unParalled_state_dict,True)

            net.to(self.device)
            net.eval()
            self.models.append(net)
            self.checkpoint_names.append(checkpoint_name)


    def predict(self, input_image, patient_info_dict):
        """
        perform the prediction given an image and the metadata.
        input_image is a ndarray read using cv2.imread(path_to_image, 1).
        note that the order of the three channels of the input_image read by cv2 is BGR.
        :param input_image: the input image to the model.
        :param patient_info_dict: a dictionary with the metadata for the given image,
        such as {'age': 52.0, 'sex': 'male', 'height': nan, 'weight': 71.3},
        where age, height and weight are of type float, while sex is of type str.
        :return: an int value indicating the class for the input image.
        """
        
        scores = []
        with torch.no_grad():
            for i in range(len(self.models)):
                checkpoint_name = self.checkpoint_names[i]
                net = self.models[i]
                img_size = 512
                image = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
                transform = T.Compose([
                    T.Resize((img_size,img_size)),
                    T.ToTensor(),
                    T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
                ])
                image = transform(image)
                image = image.unsqueeze(0)
                image = image.to(self.device, torch.float)

                score = net(image)
                # scores.append(score)
                scores += [score]

        # score = sum(scores)
        score = np.sum(scores, axis=0)

        _, pred_class = torch.max(score, 1)
        pred_class = pred_class.detach().cpu()

        return int(pred_class)

def get_model(name, num_classes=5):
    if name == 'resnet50':
        net = models.resnet50(pretrained=False)
        num_features = net.fc.in_features
        net.fc = nn.Linear(in_features=num_features, out_features=num_classes)
    elif name == 'resnet18':
    # if name == 'resnet18':
        net = models.resnet18(pretrained=False)
        num_features = net.fc.in_features
        net.fc = nn.Linear(in_features=num_features, out_features=num_classes)
    elif name == 'efficientnetb0':
        net = models.efficientnet_b0(pretrained=False)
        in_channel = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_channel, num_classes)
    elif name == 'efficientnetb1':
        net = models.efficientnet_b1(pretrained=False)
        in_channel = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_channel, num_classes)
    elif name == 'efficientnetb2':
        net = models.efficientnet_b2(pretrained=False)
        in_channel = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_channel, num_classes)
    elif name == 'efficientnetb3':
        net = models.efficientnet_b3(pretrained=False)
        in_channel = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_channel, num_classes) 
    
    else:
        raise RuntimeError("model not found")
    return net





