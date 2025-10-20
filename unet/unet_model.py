""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))



    def use_checkpointing(self):
        """启用梯度检查点以节省显存"""
        self.checkpointing = True

    def forward(self, x):
        if hasattr(self, 'checkpointing') and self.checkpointing:
            # 使用梯度检查点的前向传播
            x1 = torch.utils.checkpoint.checkpoint(self.inc, x, use_reentrant=False)
            x2 = torch.utils.checkpoint.checkpoint(self.down1, x1, use_reentrant=False)
            x3 = torch.utils.checkpoint.checkpoint(self.down2, x2, use_reentrant=False)
            x4 = torch.utils.checkpoint.checkpoint(self.down3, x3, use_reentrant=False)
            x5 = torch.utils.checkpoint.checkpoint(self.down4, x4, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self.up1, x5, x4, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self.up2, x, x3, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self.up3, x, x2, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self.up4, x, x1, use_reentrant=False)
            logits = torch.utils.checkpoint.checkpoint(self.outc, x, use_reentrant=False)
            return logits
        else:
            # 正常的前向传播
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
            return logits