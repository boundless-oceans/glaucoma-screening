import torch
from torch import nn
from torchvision import models


class ImageClassifier(nn.Module):
    def __init__(self,config,pretrained=True):
        super().__init__();name=config.get('model','efficientnet_b0')
        choices={
            'efficientnet_b0':(models.efficientnet_b0,models.EfficientNet_B0_Weights.IMAGENET1K_V1),
            'efficientnet_b2':(models.efficientnet_b2,models.EfficientNet_B2_Weights.IMAGENET1K_V1),
            'convnext_tiny':(models.convnext_tiny,models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)}
        if name not in choices: raise ValueError(f'Unknown model {name}')
        builder,weights=choices[name];self.backbone=builder(weights=weights if pretrained else None)
        dim=self.backbone.classifier[-1].in_features
        if name.startswith('convnext'):
            self.backbone.classifier[-1]=nn.Identity()
        else: self.backbone.classifier=nn.Identity()
        self.use_age=bool(config.get('use_age',False))
        self.age_head=nn.Sequential(nn.Linear(2,16),nn.ReLU(),nn.Dropout(.1)) if self.use_age else None
        self.head=nn.Sequential(nn.Dropout(.2),nn.Linear(dim+(16 if self.use_age else 0),1))

    def forward(self,images,age_features=None):
        features=self.backbone(images)
        if self.use_age:
            if age_features is None: raise ValueError('Age input required')
            features=torch.cat([features,self.age_head(age_features)],dim=1)
        return self.head(features).squeeze(1)


def build_model(config,pretrained=True):
    return ImageClassifier(config,pretrained)
