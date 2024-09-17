from resnet import get_resnet
import torch 
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, backbone, bn_momentum, pretrained=True):
        super(Net, self).__init__()
        encoder = get_resnet(backbone, momentumn=bn_momentum, pretrained=pretrained)
        self.encoder = encoder

    def forward(self, x):
        feature = self.encoder(x)
        feature = feature.view(feature.size(0), -1)
        return feature

class Classifier(nn.Module):
    def __init__(self, in_dim, classes=65):
        super(Classifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_dim, classes)

    def forward(self, feature):
        feature = self.flatten(feature)
        output = self.fc(feature)
        return output