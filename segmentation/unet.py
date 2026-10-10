import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """Attention gate for filtering encoder skip features."""

    def __init__(self, gate_channels, skip_channels, hidden_channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(gate_channels, hidden_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(hidden_channels),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(skip_channels, hidden_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(hidden_channels),
        )
        self.coefficient = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, gate_features, skip_features):
        attention = self.activation(self.gate(gate_features) + self.skip(skip_features))
        return skip_features * self.coefficient(attention)


class DoubleConv(nn.Module):
    """Two convolution blocks with an optional residual shortcut."""

    def __init__(self, in_channels, out_channels, mid_channels=None, use_residual=True):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.use_residual = use_residual
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if use_residual and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, features):
        residual = self.shortcut(features) if self.use_residual else 0
        return self.activation(self.double_conv(features) + residual)


class Down(nn.Module):
    """Max-pooling followed by a residual double convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, features):
        return self.block(features)


class Up(nn.Module):
    """Upsampling, attention-gated skip fusion, and a residual double convolution."""

    def __init__(self, in_channels, out_channels, bilinear=True, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        if use_attention:
            self.attention = AttentionBlock(
                gate_channels=in_channels // 2,
                skip_channels=in_channels // 2,
                hidden_channels=in_channels // 4,
            )

    def forward(self, decoder_features, encoder_features):
        decoder_features = self.up(decoder_features)
        diff_y = encoder_features.size(2) - decoder_features.size(2)
        diff_x = encoder_features.size(3) - decoder_features.size(3)
        decoder_features = F.pad(
            decoder_features,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        if self.use_attention:
            encoder_features = self.attention(decoder_features, encoder_features)
        return self.conv(torch.cat([encoder_features, decoder_features], dim=1))


class OutConv(nn.Module):
    """Project decoder features to the requested mask channels."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, features):
        return self.conv(features)


class UNet(nn.Module):
    """Attention-gated residual U-Net used as the SGSD teacher."""

    def __init__(self, n_channels, n_classes, bilinear=True, use_attention=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.dropout = nn.Dropout2d(p=0.3)

        self.up1 = Up(1024, 512 // factor, bilinear, use_attention)
        self.up2 = Up(512, 256 // factor, bilinear, use_attention)
        self.up3 = Up(256, 128 // factor, bilinear, use_attention)
        self.up4 = Up(128, 64, bilinear, use_attention)
        self.outc = OutConv(64, n_classes)

    def forward(self, features):
        x1 = self.inc(features)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.dropout(self.down4(x4))
        features = self.up1(x5, x4)
        features = self.up2(features, x3)
        features = self.up3(features, x2)
        features = self.up4(features, x1)
        return self.outc(features)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=1).to(device)
    sample = torch.randn(2, 3, 572, 572, device=device)
    output = model(sample)
    print(f"Input shape: {sample.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
