import torch
import torch.nn as nn
from torchvision import models
from transformers import DeiTConfig, DeiTModel, DeiTFeatureExtractor
import config

class ResNet(nn.Module):
    def __init__(self, h_dim):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        #self.to_hid = nn.Linear(512, h_dim)
    
    def forward(self, x):
        x = self.resnet(x).squeeze()
        #x = self.to_hid(x)
        return x

# TODO Implement DeiT using huggingface
class DeiT(nn.Module):
    def __init__(self, h_dim):
        super(DeiT, self).__init__()
        # self.config = DeiTConfig()
        # self.visual_encoder = DeiTModel(self.config)
        # self.feature_extractor = DeiTFeatureExtractor.from_pretrained("facebook/deit-base-distilled-patch16-224")
        self.deit_model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
        if config.train_deit is False:
            for param in self.deit_model.parameters():
                param.requires_grad = False

        self.to_hid = nn.Linear(198*768, h_dim) # This is deit output

    def forward(self, x):
        # ans = []
        # for i in range(0, x.shape[0]):
        #     inputs = self.feature_extractor(x[i, :, :, :], return_tensors="pt") #needs as a input or one image (single tensor) or a list of tensors 
        #     with torch.no_grad():
        #         outputs = self.deit_model(**inputs)[0]
        #         ans.append(outputs)
        #         # print(outputs.shape)
        # # ans = self.feature_extractor(x[0, :, :, :], return_tensors="np")
        # # print(torch.stack(ans).shape)
        # output = torch.stack(ans).squeeze()
        # return self.to_hid(output)
        # We have extracted features in the dataloader
        # features = self.feature_extractor(list(x), return_tensors="pt")["pixel_values"]
        if config.train_deit is False:
            self.deit_model.eval()
            with torch.no_grad():
                output = self.deit_model(x)[0]
        else:
            output = self.deit_model(x)[0]

        output = output.reshape(output.shape[0], -1)
        return self.to_hid(output)
    
def get_feature_extractor(model="resnet", h_dim=None):
    if model=="resnet":
        return ResNet(h_dim)
    elif model=="deit":
        return DeiT(h_dim)
    else:
        raise Exception("Feature extractor model not recognized:", model)