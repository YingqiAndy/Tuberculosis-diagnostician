from torchvision.models import resnet50, ResNet50_Weights
from torch.nn import Module, Sequential, Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from torchsummary import summary


class DeepResNet(Module):
    """
    Deep ResNet model, using pretrained ResNet50 model
    to extract features, and classification using liner
    layer after pooling.
    """
    def __init__(self, num_classes=2):
        super(DeepResNet, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features = Sequential(*list(resnet.children())[:-2])
        self.pooling = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = DeepResNet()
    summary(model, input_size=(3, 224, 224))