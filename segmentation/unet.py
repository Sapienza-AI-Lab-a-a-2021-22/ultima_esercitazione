import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConvolution(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DoubleConvolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):

    def __init__(self, input_channels=3, output_channels=1,
                 features=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder 
        for f in features:
            self.downs.append(DoubleConvolution(input_channels, f))
            input_channels = f

        # lower bottleneck layers
        self.bottleneck = DoubleConvolution(features[-1], features[-1] * 2)

        # Decoder
        for f in reversed(features):
            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(in_channels=2 * f, out_channels=f, kernel_size=3,
                              padding=1),
                ))
            self.ups.append(DoubleConvolution(2 * f, f))

        self.final_convolution = nn.Conv2d(in_channels=features[0],
                                           out_channels=output_channels,
                                           kernel_size=3, padding=1)

    def forward(self, x):
        skip_connections = list()
        for module in self.downs:
            x = module(x)
            skip_connections.append(x)
            x = self.pool(x)

        skip_connections = skip_connections[::-1]  # reverse order

        x = self.bottleneck(x)

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]
            if skip_connection.shape != x.shape:
                x = TF.resize(x, size=skip_connection.shape[2:],
                              interpolation=TF.InterpolationMode.NEAREST)
            x = torch.cat([skip_connection, x], dim=1)
            x = self.ups[i + 1](x)

        x = self.final_convolution(x)

        return x
