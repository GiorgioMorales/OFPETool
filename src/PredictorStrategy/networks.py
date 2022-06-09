from abc import ABC

import torch.nn as nn
import numpy as np
from torch import reshape, cat, Tensor, matmul, no_grad, optim, rand, index_select
from torch.autograd import Variable


class Hyper3DNetLiteReg(nn.Module, ABC):
    """Our proposed 3D-2D CNN."""

    def __init__(self, img_shape=(1, 15, 5, 5), output_size=5, output_channels=1):
        super(Hyper3DNetLiteReg, self).__init__()
        # Set stride
        stride = 1
        # If the size of the output patch is less than the input size, don't apply padding at the end
        if output_size < img_shape[2]:
            padding = 0
        else:
            padding = 1

        self.img_shape = img_shape
        self.output_size = output_size
        self.output_channels = output_channels
        nfilters = 32

        self.conv_layer1 = nn.Sequential(
            nn.Conv3d(in_channels=img_shape[0], out_channels=nfilters, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm3d(nfilters))
        self.conv_layer2 = nn.Sequential(
            nn.Conv3d(in_channels=nfilters, out_channels=nfilters, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm3d(nfilters))
        self.conv_layer3 = nn.Sequential(
            nn.Conv3d(in_channels=nfilters * 2, out_channels=nfilters, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm3d(nfilters))
        # self.drop0 = nn.Dropout(p=0.5)
        self.conv_layer4 = nn.Sequential(
            nn.Conv3d(in_channels=nfilters * 3, out_channels=nfilters, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm3d(nfilters))

        self.drop = nn.Dropout(p=0.5)

        self.sepconv1 = nn.Sequential(
            nn.Conv2d(in_channels=nfilters * 4 * img_shape[1], out_channels=nfilters * 4 * img_shape[1],
                      kernel_size=3, padding=1, groups=nfilters * 4 * img_shape[1]), nn.ReLU(),
            nn.Conv2d(in_channels=nfilters * 4 * img_shape[1], out_channels=512,
                      kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm2d(512))
        self.sepconv2 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512,
                                                kernel_size=3, padding=1, stride=stride, groups=512), nn.ReLU(),
                                      nn.Conv2d(in_channels=512, out_channels=320,
                                                kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm2d(320))
        self.drop2 = nn.Dropout(p=0.1)
        self.sepconv3 = nn.Sequential(nn.Conv2d(in_channels=320, out_channels=320,
                                                kernel_size=3, padding=1, stride=stride, groups=320), nn.ReLU(),
                                      nn.Conv2d(in_channels=320, out_channels=256,
                                                kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm2d(256))
        self.drop3 = nn.Dropout(p=0.1)
        self.sepconv4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256,
                                                kernel_size=3, padding=1, stride=stride, groups=256), nn.ReLU(),
                                      nn.Conv2d(in_channels=256, out_channels=128,
                                                kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm2d(128))
        self.sepconv5 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=32,
                                                kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32))

        # This layer is used in case the outputSize is 1
        if output_size == 1:
            self.out = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=1,
                                               kernel_size=3, padding=padding), nn.ReLU())
            self.fc = nn.Linear(9, output_channels)
        else:
            self.out = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=output_channels,
                                               kernel_size=3, padding=padding), nn.ReLU())

    def forward(self, x, device):
        # print(device)
        # 3D Feature extractor
        x = self.conv_layer1(x)
        x2 = self.conv_layer2(x)
        x = cat((x, x2), 1)
        x2 = self.conv_layer3(x)
        x = cat((x, x2), 1)
        # x = self.drop0(x)
        x2 = self.conv_layer4(x)
        x = cat((x, x2), 1)
        # Reshape 3D-2D
        x = reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))
        x = self.drop(x)
        # 2D Spatial encoder
        x = self.sepconv1(x)
        x = self.sepconv2(x)
        x = self.drop2(x)
        x = self.sepconv3(x)
        x = self.drop3(x)
        x = self.sepconv4(x)
        x = self.sepconv5(x)
        # Final output
        x = self.out(x)

        # Flatten and apply the last fc layer if the output is just a number
        if self.output_size == 1:
            x = reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
            x = self.fc(x)
        else:
            # Reshape 2D
            if self.output_channels == 1:
                x = reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))
        return x


class Russello(nn.Module, ABC):
    """Russello's 3D-CNN (2018)."""

    def __init__(self, img_shape=(1, 15, 5, 5)):
        super(Russello, self).__init__()

        self.img_shape = img_shape
        nfilters = 64
        self.conv_layer1 = nn.Sequential(
            nn.Conv3d(in_channels=img_shape[0], out_channels=nfilters, kernel_size=3, padding=1),
            nn.BatchNorm3d(nfilters), nn.ReLU())
        self.conv_layer2 = nn.Sequential(
            nn.Conv3d(in_channels=nfilters, out_channels=nfilters * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(nfilters * 2), nn.ReLU())
        self.conv_layer3 = nn.Sequential(
            nn.Conv3d(in_channels=nfilters * 2, out_channels=nfilters * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(nfilters * 4), nn.ReLU())
        self.conv_layer4 = nn.Sequential(
            nn.Conv3d(in_channels=nfilters * 4, out_channels=nfilters * 8, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(nfilters * 8), nn.ReLU())

        self.drop = nn.Dropout(p=0.2)

        if img_shape[1] == 7 or img_shape[1] == 8:
            self.fc_5 = nn.Linear(nfilters * 8 * 4 * 3 * 3, 1024)
        elif img_shape[1] == 6:
            self.fc_5 = nn.Linear(nfilters * 8 * 3 * 3 * 3, 1024)
        else:
            self.fc_5 = nn.Linear(nfilters * 8 * 5 * 3 * 3, 1024)
        self.fc_6 = nn.Linear(1024, 1)

    def forward(self, x, device):
        # 3D Feature extractor
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        # Reshape 3D-1D
        x = x.flatten(start_dim=1, end_dim=4)
        x = self.drop(x)
        # Final yield value
        x = self.fc_5(x)
        x = self.fc_6(x)
        return x


class CNNLF(nn.Module, ABC):
    """Barbosas's CNN-Late Fusion (2020)."""

    def __init__(self, img_shape=(1, 15, 5, 5)):
        super(CNNLF, self).__init__()

        self.img_shape = img_shape
        # Initialize list of parallel layers
        self.multi_stream_conv = nn.ModuleList()
        self.multi_stream_linear1 = nn.ModuleList()
        self.multi_stream_linear2 = nn.ModuleList()
        self.multi_stream_dropout = nn.ModuleList()

        for stream in range(self.img_shape[1]):
            self.multi_stream_conv.append(nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3,
                                                                  padding=1), nn.BatchNorm2d(8), nn.ReLU()))
            self.multi_stream_dropout.append(nn.Dropout(p=0.2))
            self.multi_stream_linear1.append(nn.Sequential(nn.Linear(8 * self.img_shape[2] * self.img_shape[3], 16),
                                                           nn.ReLU()))
            self.multi_stream_linear2.append(nn.Linear(16, 1))
        # Fusion step
        self.fc = nn.Linear(self.img_shape[1], 1)

    def forward(self, x, device):
        # Reshape 3D-2D
        x = x.flatten(start_dim=0, end_dim=1)
        # Process parallel streams
        streams = []
        for stream in range(self.img_shape[1]):
            # Apply convolutional layer
            conv = self.multi_stream_conv[stream](index_select(x, 1, Tensor([stream]).to(device).long()))
            # Flatten
            conv = conv.flatten(start_dim=1, end_dim=3)
            conv = self.multi_stream_dropout[stream](conv)
            # Apply FC
            linear = self.multi_stream_linear1[stream](conv)
            linear = self.multi_stream_linear2[stream](linear)
            # Store stream
            streams.append(linear)
        # Stack streams
        out = streams[0]
        for s in range(1, self.img_shape[1]):
            out = cat((out, streams[s]), dim=1)
        x = self.fc(out)
        return x


class LinearModel(nn.Module, ABC):
    """Used to train the CRF alone"""

    def __init__(self, features=10, nodes=25):
        super(LinearModel, self).__init__()
        self.features = features
        self.nodes = nodes
        # Sets learnable weights
        self.theta = None
        self.device = None
        np.random.seed(seed=7)
        self.reset_weights()

    def reset_weights(self, init=False, W=None):
        # Sets learnable weights
        if not init:
            self.theta = nn.Parameter(rand(1, self.features))
        else:
            if W is None:
                W = rand(1, self.features)
            with no_grad():
                self.theta.copy_(W)
            self.to(self.device)

    def forward(self, x, device):
        self.device = device
        B = x.shape[0]  # Number of images in the batch
        # Reshape x
        x = x.reshape(x.shape[0] * x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        # Expand Theta to match the number of elements in a batch
        theta = (self.theta.view(1,
                                 self.theta.shape[0],
                                 self.theta.shape[1])).expand(B * self.nodes, self.theta.shape[0], self.theta.shape[1])
        H = matmul(theta, x)
        # Reshape to the original shape
        H = H.reshape(B, self.nodes, 1)
        return H


# Code adapted from https://github.com/ShayanPersonal/stacked-autoencoder-pytorch
class CDAutoEncoder(nn.Module, ABC):
    """
    Denoising autoencoder layer for stacked autoencoders.
    This module is automatically trained when in model.training is True.
    Args:
        input_size: The number of features in the input
        output_size: The number of features to output
    """

    def __init__(self, input_size, output_size):
        super(CDAutoEncoder, self).__init__()

        self.forward_pass = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
        )
        self.backward_pass = nn.Sequential(
            nn.Linear(output_size, input_size),
            nn.ReLU(),
        )

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adadelta(self.parameters(), lr=1.0)

    def forward(self, x):
        # Train each autoencoder individually
        x = x.detach()

        if self.training:
            # Add noise, but use the original lossless input as the target.
            x_noisy = x * (Variable(x.data.new(x.size()).normal_(0, 0.01)) > -.1).type_as(x)
            y = self.forward_pass(x_noisy)
            x_reconstruct = self.backward_pass(y)
            loss = self.criterion(x_reconstruct, Variable(x.data, requires_grad=False))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            y = self.forward_pass(x)

        return y.detach()

    def reconstruct(self, x):
        return self.backward_pass(x)


class StackedAutoEncoder(nn.Module, ABC):
    r"""
    A stacked autoencoder made from the denoising autoencoders above.
    Each autoencoder is trained independently and at the same time.
    """

    def __init__(self, nbands):
        super(StackedAutoEncoder, self).__init__()

        self.ae1 = CDAutoEncoder(nbands, 500)
        self.ae2 = CDAutoEncoder(500, 250)
        self.ae3 = CDAutoEncoder(250, 125)
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        a1 = self.ae1(x)
        a2 = self.ae2(a1)
        a2 = self.drop(a2)
        a3 = self.ae3(a2)

        if self.training:
            return a3

        else:
            return a3, self.reconstruct(a3)

    def reconstruct(self, x):
        a2_reconstruct = self.ae3.reconstruct(x)
        a1_reconstruct = self.ae2.reconstruct(a2_reconstruct)
        x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
        return x_reconstruct
