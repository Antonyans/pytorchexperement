import re

import torch
from torch.nn.functional import softmax
from torchvision.models import resnet18
from torchvision.transforms import transforms
import torch.nn as nn

from utils import load_state_dict_from_url

from base_feature import BaseFeature

model_urls = {
    'mask': 'https://drive.google.com/uc?id=18AosQQaVDpY2PLQxOAwMTs1uaa-8mzpL',
}

class MaskDetection:
    
    def __init__(self, path="mask_weights.pt", device="cpu"):
        # super().__init__(__file__, path, device)
        super(MaskDetection, self).__init__()


    def _load_model(self, path):
        model = resnet18()
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        self.model = model

    def _create_transform(self):
        return transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def _output_postprocess(self, outputs):
        probs = softmax(outputs, 1)
        probs, masks = torch.max(probs, 1)

        response = []
        for mask, p in zip(masks, probs):
            mask_str = "masked" if mask == 0 else "unmasked"
            
            response.append({'value': mask_str, 'probability': p.item()})

        return response




def mask1(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model1 = MaskDetection(device='cpu',  **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mask'],
                                              progress=progress)
        model1.load_state_dict(state_dict)
    return model1

def _load_state_dict(model, model_url, progress):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')


def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = MaskDetection(device='cpu')
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def mask(pretrained=False, progress=True, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('mask', 32, (6, 12, 48, 32), 64, pretrained, progress,
                     **kwargs)