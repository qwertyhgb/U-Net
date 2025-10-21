"""
UNet 网络组件实现

该模块实现了UNet网络的核心组件，包括各种卷积层、下采样层、上采样层等。
这些组件是构建完整UNet网络的基础模块，每个组件都有特定的功能和作用。

组件列表：
1. DoubleConv: 双卷积层，UNet的基础构建块
2. Down: 下采样层，用于编码器路径
3. Up: 上采样层，用于解码器路径
4. OutConv: 输出卷积层，生成最终分割掩码

设计原则：
- 模块化设计：每个组件功能单一，便于复用和调试
- 标准化实现：遵循PyTorch最佳实践
- 灵活配置：支持不同的参数设置
- 性能优化：使用高效的卷积操作

技术特点：
- 使用BatchNorm提升训练稳定性
- 使用ReLU激活函数加速收敛
- 支持不同的上采样方式
- 自动处理尺寸不匹配问题
"""

# ================================
# 导入必要的模块
# ================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    双卷积层：UNet网络的基础构建块
    
    该层实现了两个连续的3x3卷积操作，是UNet网络的基本构建单元。
    每个卷积后都跟随BatchNorm和ReLU激活函数，提升训练稳定性和收敛速度。
    
    结构：Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU
    
    ================================
    参数说明：
    ================================
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        mid_channels (int, optional): 中间通道数
            - 如果未指定，使用out_channels
            - 用于控制网络宽度和参数量
    
    ================================
    设计特点：
    ================================
    1. 双卷积结构：
       - 两个3x3卷积提供更大的感受野
       - 相比单个5x5卷积，参数量更少
       - 提供更好的特征提取能力
    
    2. 标准化和激活：
       - BatchNorm加速训练收敛
       - ReLU提供非线性激活
       - inplace=True节省内存
    
    3. 参数优化：
       - bias=False配合BatchNorm使用
       - padding=1保持空间尺寸不变
       - 支持自定义中间通道数
    
    ================================
    使用示例：
    ================================
    ```python
    # 基础双卷积层
    conv = DoubleConv(64, 128)
    
    # 自定义中间通道数
    conv = DoubleConv(64, 128, mid_channels=96)
    
    # 前向传播
    x = torch.randn(1, 64, 256, 256)
    output = conv(x)
    print(output.shape)  # torch.Size([1, 128, 256, 256])
    ```
    """
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        """
        初始化双卷积层
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            mid_channels: 中间通道数，默认为out_channels
        """
        super().__init__()
        
        # 如果没有指定中间通道数，使用输出通道数
        if not mid_channels:
            mid_channels = out_channels
        
        # 构建双卷积序列
        # 使用Sequential容器将多个层组合在一起
        self.double_conv = nn.Sequential(
            # 第一个卷积层：输入 -> 中间通道
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),  # 批归一化，加速收敛
            nn.ReLU(inplace=True),         # ReLU激活，inplace节省内存
            
            # 第二个卷积层：中间通道 -> 输出通道
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # 批归一化
            nn.ReLU(inplace=True)          # ReLU激活
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        Args:
            x: 输入张量，形状为(batch_size, in_channels, height, width)
            
        Returns:
            输出张量，形状为(batch_size, out_channels, height, width)
        """
        return self.double_conv(x)


class Down(nn.Module):
    """
    下采样层：用于UNet编码器路径
    
    该层实现了下采样操作，通过最大池化减少空间尺寸，然后使用双卷积提取特征。
    这是UNet编码器路径的核心组件，用于逐步提取更高级别的特征。
    
    结构：MaxPool2d -> DoubleConv
    
    ================================
    参数说明：
    ================================
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
    
    ================================
    设计特点：
    ================================
    1. 下采样操作：
       - 使用2x2最大池化，空间尺寸减半
       - 保留最显著的特征，丢弃细节信息
       - 增加感受野，提取更高级别特征
    
    2. 特征提取：
       - 下采样后使用双卷积进一步提取特征
       - 增加通道数，提升特征表达能力
       - 保持空间尺寸不变
    
    3. 信息压缩：
       - 空间信息压缩：H×W -> H/2×W/2
       - 通道信息扩展：in_channels -> out_channels
       - 总体信息量保持平衡
    
    ================================
    使用示例：
    ================================
    ```python
    # 创建下采样层
    down = Down(64, 128)
    
    # 前向传播
    x = torch.randn(1, 64, 256, 256)
    output = down(x)
    print(output.shape)  # torch.Size([1, 128, 128, 128])
    ```
    
    ================================
    注意事项：
    ================================
    1. 尺寸要求：
       - 输入尺寸必须是偶数
       - 输出尺寸为输入尺寸的一半
       - 建议使用2的幂次方尺寸
    
    2. 信息损失：
       - 最大池化会丢失部分空间信息
       - 这是编码器路径的必然结果
       - 通过跳跃连接可以部分恢复
    
    3. 特征层次：
       - 每层提取不同尺度的特征
       - 深层特征更加抽象和语义化
       - 浅层特征更加细节和局部化
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        初始化下采样层
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super().__init__()
        
        # 构建下采样序列：最大池化 + 双卷积
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),                           # 2x2最大池化，尺寸减半
            DoubleConv(in_channels, out_channels)      # 双卷积特征提取
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        Args:
            x: 输入张量，形状为(batch_size, in_channels, height, width)
            
        Returns:
            输出张量，形状为(batch_size, out_channels, height/2, width/2)
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    上采样层：用于UNet解码器路径
    
    该层实现了上采样操作，通过双线性插值或转置卷积增加空间尺寸，
    然后与编码器特征进行跳跃连接，最后使用双卷积融合特征。
    这是UNet解码器路径的核心组件，用于恢复空间分辨率和融合多尺度特征。
    
    结构：Upsample/ConvTranspose -> Skip Connection -> DoubleConv
    
    ================================
    参数说明：
    ================================
    Args:
        in_channels (int): 输入通道数（来自解码器）
        out_channels (int): 输出通道数
        bilinear (bool): 是否使用双线性插值上采样
            - True: 使用双线性插值，参数量少，速度慢
            - False: 使用转置卷积，参数量多，速度快
    
    ================================
    设计特点：
    ================================
    1. 上采样方式：
       - 双线性插值：参数量少，但学习能力有限
       - 转置卷积：参数量多，但学习能力更强
       - 根据需求选择合适的上采样方式
    
    2. 跳跃连接：
       - 将编码器特征与解码器特征融合
       - 保留细节信息，提升边界分割精度
       - 自动处理尺寸不匹配问题
    
    3. 特征融合：
       - 通过通道拼接融合不同层级的特征
       - 使用双卷积进一步提取和融合特征
       - 生成高质量的分割结果
    
    ================================
    使用示例：
    ================================
    ```python
    # 使用双线性插值上采样
    up = Up(1024, 512, bilinear=True)
    
    # 使用转置卷积上采样
    up = Up(1024, 512, bilinear=False)
    
    # 前向传播
    x1 = torch.randn(1, 1024, 32, 32)  # 解码器特征
    x2 = torch.randn(1, 512, 64, 64)   # 编码器特征
    output = up(x1, x2)
    print(output.shape)  # torch.Size([1, 512, 64, 64])
    ```
    
    ================================
    注意事项：
    ================================
    1. 尺寸匹配：
       - 自动处理尺寸不匹配问题
       - 使用padding确保尺寸一致
       - 支持任意尺寸的输入
    
    2. 内存占用：
       - 跳跃连接会增加内存占用
       - 大图像可能需要较多显存
       - 可以使用梯度检查点优化
    
    3. 特征质量：
       - 跳跃连接保留细节信息
       - 双卷积进一步提取特征
       - 生成高质量的分割结果
    """
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        """
        初始化上采样层
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            bilinear: 是否使用双线性插值上采样
        """
        super().__init__()
        
        # 根据上采样方式选择不同的实现
        if bilinear:
            # ================================
            # 双线性插值上采样
            # ================================
            # 使用双线性插值进行上采样，参数量少但学习能力有限
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 由于上采样不改变通道数，需要调整双卷积的输入通道数
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # ================================
            # 转置卷积上采样
            # ================================
            # 使用转置卷积进行上采样，参数量多但学习能力更强
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # 转置卷积会改变通道数，直接使用原始通道数
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        Args:
            x1: 解码器特征，来自上一层的上采样结果
            x2: 编码器特征，来自对应层的跳跃连接
            
        Returns:
            融合后的特征，形状为(batch_size, out_channels, height, width)
        """
        # ================================
        # 1. 上采样解码器特征
        # ================================
        # 将解码器特征上采样到与编码器特征相同的尺寸
        x1 = self.up(x1)
        
        # ================================
        # 2. 处理尺寸不匹配问题
        # ================================
        # 计算两个特征图在高度和宽度上的差异
        # 由于池化和上采样可能产生尺寸不匹配，需要手动调整
        diffY = x2.size()[2] - x1.size()[2]  # 高度差异
        diffX = x2.size()[3] - x1.size()[3]  # 宽度差异
        
        # 使用padding调整x1的尺寸，使其与x2匹配
        # padding格式：[left, right, top, bottom]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # ================================
        # 3. 跳跃连接：特征融合
        # ================================
        # 在通道维度上拼接两个特征图
        # x2是编码器特征（细节信息），x1是解码器特征（语义信息）
        x = torch.cat([x2, x1], dim=1)
        
        # ================================
        # 4. 特征提取和融合
        # ================================
        # 使用双卷积进一步提取和融合特征
        return self.conv(x)


class OutConv(nn.Module):
    """
    输出卷积层：生成最终的分割掩码
    
    该层是UNet网络的最后一层，使用1x1卷积将特征图转换为分割掩码。
    这是整个网络的输出层，负责生成每个像素的类别预测。
    
    结构：Conv2d(1x1) -> 分割掩码
    
    ================================
    参数说明：
    ================================
    Args:
        in_channels (int): 输入通道数（来自最后一层解码器）
        out_channels (int): 输出通道数（分割类别数）
    
    ================================
    设计特点：
    ================================
    1. 1x1卷积：
       - 不改变空间尺寸，只改变通道数
       - 参数量少，计算效率高
       - 适合作为输出层
    
    2. 类别映射：
       - 将特征图映射到类别空间
       - 每个通道对应一个类别
       - 输出logits，需要后续激活
    
    3. 最终输出：
       - 生成每个像素的类别预测
       - 形状：(batch_size, n_classes, height, width)
       - 内容：每个像素的类别概率或logits
    
    ================================
    使用示例：
    ================================
    ```python
    # 创建输出卷积层
    out_conv = OutConv(64, 2)  # 二分类任务
    
    # 前向传播
    x = torch.randn(1, 64, 256, 256)
    output = out_conv(x)
    print(output.shape)  # torch.Size([1, 2, 256, 256])
    
    # 应用激活函数
    if n_classes == 1:
        output = torch.sigmoid(output)  # 二分类
    else:
        output = torch.softmax(output, dim=1)  # 多分类
    ```
    
    ================================
    注意事项：
    ================================
    1. 输出处理：
       - 输出是logits，需要激活函数
       - 二分类使用sigmoid激活
       - 多分类使用softmax激活
    
    2. 类别数：
       - 输出通道数等于类别数
       - 包括背景类在内
       - 确保与训练时一致
    
    3. 空间尺寸：
       - 输出尺寸与输入尺寸相同
       - 每个像素都有类别预测
       - 支持任意尺寸的输入
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        初始化输出卷积层
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数（分割类别数）
        """
        super(OutConv, self).__init__()
        
        # 使用1x1卷积将特征图转换为分割掩码
        # 1x1卷积不改变空间尺寸，只改变通道数
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        Args:
            x: 输入特征图，形状为(batch_size, in_channels, height, width)
            
        Returns:
            分割掩码logits，形状为(batch_size, out_channels, height, width)
        """
        return self.conv(x)
