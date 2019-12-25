import torch
import numpy as np
import torch.nn.functional as F
import torchvision.utils as vutils
from models.generators.resnet256 import ResNetGenerator
import random
import utils
import matplotlib.pyplot as plt

weight_path = 'gen_latest.pth.tar'
num_classes = 100
device = torch.device('cpu')

gen_weight = torch.load(weight_path)
gen = ResNetGenerator(
            64,
            128,
            4,
            activation=F.relu,
            num_classes=num_classes,
            distribution='normal'
        )
gen.load_state_dict(state_dict=gen_weight['model'])

z = utils.sample_z(
    16, 128, deivce, 'normal'
)
for i in range(10000):
    add_z = utils.sample_z(
        16, 128, device, 'normal'
    )
    if i % 100 == 0:
        c = utils.sample_pseudo_labels(num_classes, 1, device)

    z += add_z * random.random() * 0.1
    fake = gen(z, c)
    vutils.save_images(
            fake,
            'restuls/demo/{}.jpg'.format(i),
            n_rows=1,
            normalize=True)
