import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import models


class UpConv(nn.Module):

    def __init__(self, in_ch, out_ch, scale=2):
        super(UpConv, self).__init__()
        neck_ch = in_ch // 4
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=scale),
            nn.Conv2d(in_channels=in_ch, out_channels=neck_ch,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(neck_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=neck_ch, out_channels=neck_ch,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(neck_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=neck_ch, out_channels=out_ch,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.up_conv(x)
        return x
    

class BottleneckConv(nn.Module):
    
    def __init__(self, in_ch, out_ch, stride=1):
        super(BottleneckConv, self).__init__()
        neck_ch = in_ch // 4
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=neck_ch,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(neck_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=neck_ch, out_channels=neck_ch,
                      kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(neck_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=neck_ch, out_channels=out_ch,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x


class UNet50(nn.Module):
    """
    TODO
    """

    def __init__(self, output_channels=1):
        """
        TODO: Rendere modulare il decoder
        :param use_gpu:
        """
        super(UNet50, self).__init__()
        # ==================================================================
        # ENCODER
        # ==================================================================
        # extract layers from resnet50
        base_model = models.resnet50(pretrained=True)
        self.initial_conv = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu
        )
        self.pool = base_model.maxpool

        self.encoder_list = nn.ModuleList()
        for layer in [base_model.layer1, base_model.layer2, base_model.layer3, base_model.layer4]:
            self.encoder_list.append(layer)

        self.bottleneck = BottleneckConv(2048, 4096, stride=2)
            
        # ==================================================================
        # DECODER
        # ==================================================================
        self.decoder_list = nn.ModuleList()
        # decoder layers
        features = [2048, 1024, 512, 256]
        for i in range(len(features)):
            self.decoder_list.append(UpConv(features[i]*2, features[i]))
            self.decoder_list.append(BottleneckConv(features[i] * 2, features[i]))
        self.decoder_list.append(UpConv(256, 64))
        self.decoder_list.append(BottleneckConv(128, 64))
        
        # final upsampling and convolutions
        self.final_conv =  nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, output_channels, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip_connections = list()
        # encoding
        x = self.initial_conv(x)
        skip_connections.append(x)
        x = self.pool(x)

        for layer in self.encoder_list:
            x = layer(x)
            skip_connections.append(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # reverse the order

        # decoding
        for i in range(0, len(self.decoder_list), 2):
            x = self.decoder_list[i](x)
            skip_connection = skip_connections[i//2]
            if skip_connection.shape != x.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            x = torch.cat([skip_connection, x], dim=1)
            x = self.decoder_list[i+1](x)
             
        x = self.final_conv(x)
        # x = self.sigmoid(x)

        return x
