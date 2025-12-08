import torch.nn as nn
from torchvision.models import resnet18 as tv_resnet18
from torchvision.models import ResNet18_Weights


def resnet18(pretrained: bool = False):
    """
    Wrapper around torchvision's ResNet18 that adapts the classifier head
    to the 10 CIFAR classes. Optionally loads ImageNet pretrained weights.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = tv_resnet18(weights=weights)

    # Replace final fully-connected layer to produce 10 logits
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model




