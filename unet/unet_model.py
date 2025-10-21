"""
UNet 完整网络架构实现

该模块实现了经典的UNet编码器-解码器分割网络架构。
UNet是一种专门用于图像分割的卷积神经网络，在医学图像分割领域表现优异。

网络特点：
1. 编码器-解码器结构：编码器提取特征，解码器恢复空间分辨率
2. 跳跃连接：将编码器特征直接传递到解码器，保留细节信息
3. 对称设计：编码器和解码器层数相同，结构对称
4. 多尺度特征：通过不同层级的特征融合提升分割精度

技术优势：
- 跳跃连接保留细节信息，提升边界分割精度
- 编码器-解码器结构适合像素级分类任务
- 对称设计便于理解和调试
- 可扩展性强，支持不同输入尺寸和类别数

适用场景：
- 医学图像分割（器官、病变区域等）
- 卫星图像分割（建筑、道路等）
- 生物图像分析（细胞分割等）
- 自动驾驶场景分割
- 工业检测和质量控制

网络结构：
输入 -> 编码器 -> 瓶颈层 -> 解码器 -> 输出
  |      |         |         |        |
  |      |         |         |        v
  |      |         |         |    分类头
  |      |         |         |        |
  |      |         |         |        v
  |      |         |         |    分割掩码
  |      |         |         |
  |      |         |         v
  |      |         |    上采样+特征融合
  |      |         |
  |      |         v
  |      |     瓶颈层(最深层)
  |      |
  |      v
  |   下采样+特征提取
  |
  v
输入图像

作者：Ronneberger et al. (2015)
论文：U-Net: Convolutional Networks for Biomedical Image Segmentation
"""

# ================================
# 导入必要的模块
# ================================
import torch
import torch.nn as nn
from .unet_parts import *


class UNet(nn.Module):
    """
    UNet 编码器-解码器分割网络
    
    该类实现了经典的UNet架构，专门用于图像分割任务。
    采用编码器-解码器结构，通过跳跃连接保留细节信息。
    
    ================================
    网络架构说明：
    ================================
    编码器部分（下采样）：
    - inc: 输入卷积层 (n_channels -> 64)
    - down1: 下采样层1 (64 -> 128)
    - down2: 下采样层2 (128 -> 256)
    - down3: 下采样层3 (256 -> 512)
    - down4: 下采样层4 (512 -> 1024)
    
    解码器部分（上采样）：
    - up1: 上采样层1 (1024 -> 512)
    - up2: 上采样层2 (512 -> 256)
    - up3: 上采样层3 (256 -> 128)
    - up4: 上采样层4 (128 -> 64)
    - outc: 输出卷积层 (64 -> n_classes)
    
    跳跃连接：
    - 编码器每层的特征图直接传递到对应的解码器层
    - 保留细节信息，提升边界分割精度
    
    ================================
    参数说明：
    ================================
    Args:
        n_channels (int): 输入图像通道数
            - 3: RGB图像
            - 1: 灰度图像
            - 4: RGBA图像
        
        n_classes (int): 分割类别数
            - 2: 二分类（前景/背景）
            - 3+: 多分类分割
            - 包括背景类在内的总类别数
        
        bilinear (bool): 是否使用双线性插值上采样
            - True: 使用双线性插值，参数量少，速度慢
            - False: 使用转置卷积，参数量多，速度快
    
    ================================
    网络特点：
    ================================
    1. 对称设计：
       - 编码器和解码器层数相同
       - 每层特征图尺寸对应
       - 便于理解和调试
    
    2. 跳跃连接：
       - 保留编码器的细节信息
       - 提升边界分割精度
       - 缓解梯度消失问题
    
    3. 多尺度特征：
       - 不同层级提取不同尺度特征
       - 全局和局部信息结合
       - 提升分割性能
    
    4. 灵活配置：
       - 支持不同输入尺寸
       - 支持不同类别数
       - 支持不同上采样方式
    
    ================================
    使用示例：
    ================================
    ```python
    # 创建二分类UNet模型
    model = UNet(n_channels=3, n_classes=2, bilinear=False)
    
    # 创建多分类UNet模型
    model = UNet(n_channels=3, n_classes=5, bilinear=True)
    
    # 前向传播
    input_tensor = torch.randn(1, 3, 256, 256)
    output = model(input_tensor)
    print(output.shape)  # torch.Size([1, n_classes, 256, 256])
    ```
    
    ================================
    注意事项：
    ================================
    1. 输入尺寸：
       - 建议使用2的幂次方尺寸（如256, 512, 1024）
       - 避免使用奇数尺寸，可能影响上采样
       - 最小尺寸建议64x64以上
    
    2. 内存占用：
       - 大图像需要较多显存
       - 可以通过减小输入尺寸或使用梯度检查点优化
       - 建议在GPU上运行
    
    3. 训练技巧：
       - 使用数据增强提升泛化能力
       - 使用混合精度训练节省显存
       - 使用学习率调度器提升收敛性能
    """
    
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = False):
        """
        初始化UNet网络
        
        Args:
            n_channels: 输入图像通道数
            n_classes: 分割类别数
            bilinear: 是否使用双线性插值上采样
        """
        super(UNet, self).__init__()
        
        # ================================
        # 网络配置参数
        # ================================
        self.n_channels = n_channels      # 输入通道数
        self.n_classes = n_classes         # 输出类别数
        self.bilinear = bilinear          # 上采样方式
        
        # ================================
        # 编码器部分（下采样路径）
        # ================================
        # 输入卷积层：将输入图像转换为64通道特征图
        # 使用两个3x3卷积层，保持空间尺寸不变
        self.inc = DoubleConv(n_channels, 64)
        
        # 下采样层1：64 -> 128通道，空间尺寸减半
        # 包含最大池化和双卷积操作
        self.down1 = Down(64, 128)
        
        # 下采样层2：128 -> 256通道，空间尺寸减半
        self.down2 = Down(128, 256)
        
        # 下采样层3：256 -> 512通道，空间尺寸减半
        self.down3 = Down(256, 512)
        
        # 下采样层4：512 -> 1024通道，空间尺寸减半
        # 这是最深层，特征图尺寸最小，感受野最大
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # ================================
        # 解码器部分（上采样路径）
        # ================================
        # 上采样层1：1024 -> 512通道，空间尺寸翻倍
        # 使用跳跃连接融合编码器特征
        self.up1 = Up(1024, 512 // factor, bilinear)
        
        # 上采样层2：512 -> 256通道，空间尺寸翻倍
        self.up2 = Up(512, 256 // factor, bilinear)
        
        # 上采样层3：256 -> 128通道，空间尺寸翻倍
        self.up3 = Up(256, 128 // factor, bilinear)
        
        # 上采样层4：128 -> 64通道，空间尺寸翻倍
        self.up4 = Up(128, 64, bilinear)
        
        # 输出卷积层：64 -> n_classes通道
        # 使用1x1卷积生成最终的分割掩码
        self.outc = OutConv(64, n_classes)



    def use_checkpointing(self):
        """
        启用梯度检查点机制以节省显存
        
        梯度检查点是一种内存优化技术，通过重新计算中间激活值来减少显存占用。
        这会用计算时间换取显存空间，适用于大模型或显存不足的情况。
        
        工作原理：
        1. 在前向传播时不保存中间激活值
        2. 在反向传播时重新计算需要的激活值
        3. 显存占用减少约50%，但计算时间增加约20%
        
        适用场景：
        - 显存不足时训练大模型
        - 处理高分辨率图像
        - 大批次训练时
        
        注意事项：
        - 会增加训练时间
        - 需要确保模型支持梯度检查点
        - 某些操作可能不支持检查点
        """
        self.checkpointing = True
        logging.info('已启用梯度检查点机制，显存占用将减少约50%')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        UNet前向传播函数
        
        该函数实现了UNet网络的完整前向传播过程，包括编码器、解码器和跳跃连接。
        支持梯度检查点机制，可以在显存不足时启用以节省内存。
        
        ================================
        前向传播流程：
        ================================
        1. 编码器路径（下采样）：
           - 输入图像 -> 64通道特征图
           - 64 -> 128通道，尺寸减半
           - 128 -> 256通道，尺寸减半
           - 256 -> 512通道，尺寸减半
           - 512 -> 1024通道，尺寸减半
        
        2. 解码器路径（上采样）：
           - 1024 -> 512通道，尺寸翻倍 + 跳跃连接
           - 512 -> 256通道，尺寸翻倍 + 跳跃连接
           - 256 -> 128通道，尺寸翻倍 + 跳跃连接
           - 128 -> 64通道，尺寸翻倍 + 跳跃连接
        
        3. 输出层：
           - 64 -> n_classes通道，生成分割掩码
        
        ================================
        参数说明：
        ================================
        Args:
            x (torch.Tensor): 输入图像张量
                - 形状: (batch_size, n_channels, height, width)
                - 数据类型: torch.float32
                - 值范围: [0, 1]（归一化后）
        
        Returns:
            torch.Tensor: 分割预测结果
                - 形状: (batch_size, n_classes, height, width)
                - 数据类型: torch.float32
                - 内容: 每个像素的类别概率或logits
        
        ================================
        跳跃连接说明：
        ================================
        跳跃连接是UNet的核心特性，将编码器的特征图直接传递到解码器：
        
        编码器层 -> 解码器层
        x1 (64通道)  -> up4 (与x1融合)
        x2 (128通道) -> up3 (与x2融合)
        x3 (256通道) -> up2 (与x3融合)
        x4 (512通道) -> up1 (与x4融合)
        
        作用：
        - 保留细节信息，提升边界分割精度
        - 缓解梯度消失问题
        - 提供多尺度特征融合
        
        ================================
        使用示例：
        ================================
        ```python
        # 创建模型
        model = UNet(n_channels=3, n_classes=2)
        
        # 前向传播
        input_tensor = torch.randn(1, 3, 256, 256)
        output = model(input_tensor)
        print(output.shape)  # torch.Size([1, 2, 256, 256])
        
        # 启用梯度检查点
        model.use_checkpointing()
        output = model(input_tensor)
        ```
        
        ================================
        注意事项：
        ================================
        1. 输入尺寸：
           - 建议使用2的幂次方尺寸
           - 最小尺寸64x64以上
           - 避免奇数尺寸
        
        2. 内存管理：
           - 大图像需要较多显存
           - 可以使用梯度检查点优化
           - 建议在GPU上运行
        
        3. 输出处理：
           - 二分类任务：使用sigmoid激活
           - 多分类任务：使用softmax激活
           - 需要后处理得到最终掩码
        """
        if hasattr(self, 'checkpointing') and self.checkpointing:
            # ================================
            # 使用梯度检查点的前向传播
            # ================================
            # 编码器路径：逐层下采样，提取多尺度特征
            x1 = torch.utils.checkpoint.checkpoint(self.inc, x, use_reentrant=False)
            x2 = torch.utils.checkpoint.checkpoint(self.down1, x1, use_reentrant=False)
            x3 = torch.utils.checkpoint.checkpoint(self.down2, x2, use_reentrant=False)
            x4 = torch.utils.checkpoint.checkpoint(self.down3, x3, use_reentrant=False)
            x5 = torch.utils.checkpoint.checkpoint(self.down4, x4, use_reentrant=False)
            
            # 解码器路径：逐层上采样，融合编码器特征
            x = torch.utils.checkpoint.checkpoint(self.up1, x5, x4, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self.up2, x, x3, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self.up3, x, x2, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self.up4, x, x1, use_reentrant=False)
            
            # 输出层：生成最终的分割掩码
            logits = torch.utils.checkpoint.checkpoint(self.outc, x, use_reentrant=False)
            return logits
        else:
            # ================================
            # 正常的前向传播（无梯度检查点）
            # ================================
            # 编码器路径：逐层下采样，提取多尺度特征
            # 每层都保存特征图，用于后续的跳跃连接
            x1 = self.inc(x)        # 输入层：n_channels -> 64
            x2 = self.down1(x1)     # 下采样1：64 -> 128
            x3 = self.down2(x2)    # 下采样2：128 -> 256
            x4 = self.down3(x3)    # 下采样3：256 -> 512
            x5 = self.down4(x4)    # 下采样4：512 -> 1024（瓶颈层）
            
            # 解码器路径：逐层上采样，融合编码器特征
            # 通过跳跃连接将编码器特征融合到解码器
            x = self.up1(x5, x4)    # 上采样1：1024 -> 512，融合x4
            x = self.up2(x, x3)     # 上采样2：512 -> 256，融合x3
            x = self.up3(x, x2)     # 上采样3：256 -> 128，融合x2
            x = self.up4(x, x1)     # 上采样4：128 -> 64，融合x1
            
            # 输出层：生成最终的分割掩码
            # 使用1x1卷积将64通道特征图转换为n_classes通道
            logits = self.outc(x)
            return logits