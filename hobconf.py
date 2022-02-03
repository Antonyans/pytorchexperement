dependencies = ['torch']

from main import mask
from torch.nn.functional import softmax
from torchvision.models import resnet18
from torchvision.transforms import transforms