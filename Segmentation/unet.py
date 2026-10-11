import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """attention gate"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DoubleConv(nn.Module):
    """(conv -> bn -> silu) x2 + residual"""
    def __init__(self, in_channels, out_channels, mid_channels=None, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Sequential()
        if self.use_residual:
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        
        self.final_act = nn.SiLU(inplace=True)

    def forward(self, x):
        if self.use_residual:
            return self.final_act(self.double_conv(x) + self.shortcut(x))
        else:
            return self.final_act(self.double_conv(x))

class Down(nn.Module):
    """downsample: maxpool + DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    upsampling block
    args:
        up_in   : input channels before upsampling (from the deeper stage)
        skip_in : channels of the skip connection (from the matching encoder stage)
        out     : output channels of this block
        bilinear: True for bilinear interpolation, False for transposed conv
        use_attention: whether to gate the skip connection
    """
    def __init__(self, up_in, skip_in, out_channels, bilinear=True, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            up_out = up_in          # channel count unchanged after upsampling
        else:
            self.up = nn.ConvTranspose2d(up_in, up_in // 2, kernel_size=2, stride=2)
            up_out = up_in // 2

        # total channels after concat
        concat_channels = skip_in + up_out

        # DoubleConv takes the concatenated channels and emits out_channels
        self.conv = DoubleConv(concat_channels, out_channels)

        if self.use_attention:
            # attention: g from the upsampling branch (up_out), x from the skip (skip_in)
            F_int = max(up_out, skip_in) // 2
            self.att = AttentionBlock(F_g=up_out, F_l=skip_in, F_int=F_int)

    def forward(self, x1, x2):
        # x1: deep features to upsample, x2: skip from the same stage
        x1 = self.up(x1)

        # pad to match sizes when padding made them differ
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        if self.use_attention:
            x2 = self.att(g=x1, x=x2)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """output head: 1x1 conv"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, use_attention=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # encoder, unchanged
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)   # emits 512 when bilinear=True

        self.dropout = nn.Dropout2d(p=0.3)

        # ---------- decoder, redesigned so up4 emits 768 ----------
        # output channels per upsampling block, decreasing
        C1 = 1024   # up1 out
        C2 = 896    # up2 out
        C3 = 832    # up3 out
        C4 = 768    # up4 out -- the target, matches vit-b embed_dim

        # up1: from down4 (512) + skip x4 (512)
        self.up1 = Up(up_in=512, skip_in=512, out_channels=C1,
                      bilinear=bilinear, use_attention=use_attention)
        # up2: from up1 (C1) + skip x3 (256)
        self.up2 = Up(up_in=C1, skip_in=256, out_channels=C2,
                      bilinear=bilinear, use_attention=use_attention)
        # up3: from up2 (C2) + skip x2 (128)
        self.up3 = Up(up_in=C2, skip_in=128, out_channels=C3,
                      bilinear=bilinear, use_attention=use_attention)
        # up4: from up3 (C3) + skip x1 (64)
        self.up4 = Up(up_in=C3, skip_in=64, out_channels=C4,
                      bilinear=bilinear, use_attention=use_attention)

        # output head takes up4's channel count C4
        self.outc = OutConv(C4, n_classes)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)          # 64
        x2 = self.down1(x1)       # 128
        x3 = self.down2(x2)       # 256
        x4 = self.down3(x3)       # 512
        x5 = self.down4(x4)       # 512 (bilinear=True)

        x5 = self.dropout(x5)

        # decoder
        x = self.up1(x5, x4)      # -> C1=1024
        x = self.up2(x, x3)       # -> C2=896
        x = self.up3(x, x2)       # -> C3=832
        x = self.up4(x, x1)       # -> C4=768

        logits = self.outc(x)     # -> n_classes
        return logits

# ============= example =============
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=1, bilinear=True, use_attention=True).to(device)
    
    dummy_input = torch.randn(2, 3, 572, 572).to(device)
    output = model(dummy_input)
    
    print(f"input shape: {dummy_input.shape}")
    print(f"output shape: {output.shape}")   # torch.Size([2, 1, 572, 572])
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")