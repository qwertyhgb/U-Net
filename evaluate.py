"""
模型评估模块

该模块用于在验证集上评估训练好的U-Net模型性能，使用Dice系数作为评价指标。
Dice系数是医学图像分割中常用的评价指标，衡量预测结果与真实标签的重叠程度。

主要功能：
1. 在验证集上进行模型推理
2. 计算Dice系数评估分割质量
3. 支持二分类和多分类任务
4. 使用自动混合精度加速推理
5. 提供详细的进度显示

Dice系数计算公式：
Dice = 2 * |A ∩ B| / (|A| + |B|)
其中A是预测结果，B是真实标签，值越接近1表示分割效果越好。
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()  # 装饰器：禁用梯度计算，节省内存并加速推理
def evaluate(net, dataloader, device, amp):
    """
    在验证集上评估模型性能
    
    该函数执行完整的模型评估流程：
    1. 将模型设置为评估模式
    2. 遍历验证集的所有批次
    3. 对每个批次进行前向推理
    4. 计算Dice系数得分
    5. 返回平均得分
    
    Args:
        net (torch.nn.Module): 训练好的U-Net模型实例
            - 必须是UNet类的实例，包含n_classes属性
            - 模型应该已经加载了训练好的权重
        dataloader (torch.utils.data.DataLoader): 验证集数据加载器
            - 每个批次包含'image'和'mask'两个键
            - 'image': 输入图像张量，形状为(batch_size, channels, height, width)
            - 'mask': 真实分割标签，形状为(batch_size, height, width)
        device (torch.device): 计算设备
            - 支持'cuda'、'cpu'、'mps'等设备类型
            - 用于将数据移动到相应的计算设备上
        amp (bool): 是否启用自动混合精度训练
            - True: 使用半精度浮点数(FP16)加速推理，减少显存占用
            - False: 使用全精度浮点数(FP32)，确保数值稳定性
    
    Returns:
        float: 验证集上的平均Dice系数得分
            - 取值范围：[0, 1]
            - 1表示完美分割，0表示完全错误
            - 通常认为Dice > 0.7是较好的分割效果
    
    Raises:
        AssertionError: 当真实标签的值超出预期范围时抛出
            - 二分类任务：标签值应在[0, 1]范围内
            - 多分类任务：标签值应在[0, n_classes)范围内
    
    Example:
        >>> model = UNet(n_channels=3, n_classes=2)
        >>> model.load_state_dict(torch.load('best_model.pth'))
        >>> val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        >>> dice_score = evaluate(model, val_loader, torch.device('cuda'), amp=True)
        >>> print(f'验证集Dice得分: {dice_score:.4f}')
    """
    # 将模型设置为评估模式
    # 这会关闭dropout、batch normalization的训练模式等训练时特有的行为
    # 确保推理时模型行为的一致性
    net.eval()
    
    # 获取验证集的批次数量，用于计算平均得分和显示进度
    num_val_batches = len(dataloader)
    
    # 初始化Dice得分累加器
    # 将对所有批次的Dice得分进行累加，最后计算平均值
    dice_score = 0

    # 使用自动混合精度进行推理（如果启用）
    # autocast会自动选择合适的精度进行计算，在保持数值稳定性的同时提升性能
    # 对于MPS设备需要特殊处理，因为Apple Silicon的MPS对autocast支持有限
    # 因此当设备类型为'mps'时，使用'cpu'的autocast设置
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        # 遍历验证集的所有批次数据
        # tqdm提供进度条显示，方便监控评估进度
        # leave=False表示评估完成后不保留进度条
        for batch in tqdm(dataloader, total=num_val_batches, desc='验证轮次', unit='batch', leave=False):
            # 从批次中提取图像和真实标签
            # batch是一个字典，包含'image'和'mask'两个键
            image, mask_true = batch['image'], batch['mask']

            # 将图像和真实标签移动到指定设备上，并转换为适当的类型
            # image: 转换为float32类型，使用channels_last内存格式优化性能
            # mask_true: 转换为long类型，因为标签通常是整数索引
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # 模型前向传播，得到预测结果
            # 输出是原始logits，形状为(batch_size, n_classes, height, width)
            # 对于二分类任务，n_classes=1；对于多分类任务，n_classes>1
            mask_pred = net(image)

            # 根据分类任务类型计算Dice系数
            if net.n_classes == 1:
                # 二分类任务：前景/背景分割
                # 验证真实标签的值是否在有效范围内[0, 1]
                assert mask_true.min() >= 0 and mask_true.max() <= 1, '真实掩码的索引应该在[0, 1]范围内'
                
                # 将logits转换为概率并应用阈值进行二值化
                # sigmoid将输出映射到[0,1]区间，阈值0.5用于二值化决策
                # 结果形状：(batch_size, 1, height, width)
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                
                # 计算二分类Dice系数得分
                # reduce_batch_first=False表示不先对批次维度进行归约
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                # 多分类任务：多个类别的前景分割
                # 验证真实标签的值是否在有效范围内[0, n_classes)
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, '真实掩码的索引应该在[0, n_classes]范围内'
                
                # 将真实标签转换为one-hot编码格式
                # 确保mask_true是正确的维度：(batch_size, height, width)
                if mask_true.dim() == 4:  # 如果是4维，去掉通道维度
                    mask_true_for_onehot = mask_true.squeeze(1)
                else:
                    mask_true_for_onehot = mask_true
                
                # one_hot: (batch_size, height, width) -> (batch_size, height, width, n_classes)
                # permute(0, 3, 1, 2): 调整维度顺序为(batch_size, n_classes, height, width)
                mask_true = F.one_hot(mask_true_for_onehot, net.n_classes).permute(0, 3, 1, 2).float()
                
                # 将预测结果转换为one-hot编码格式
                # argmax(dim=1): 在类别维度上取最大值的索引，得到预测类别
                # 然后转换为one-hot编码，形状与mask_true相同
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                
                # 计算多类Dice系数得分，忽略背景类（第0类）
                # [:, 1:]表示从第1类开始（跳过背景类），只计算前景类别的Dice得分
                # 这样可以更准确地评估模型对感兴趣区域的分割能力
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    # 将模型恢复为训练模式
    # 虽然当前函数是评估函数，但恢复训练模式是一个良好的编程实践
    # 确保模型在后续使用时的行为一致性
    net.train()
    
    # 返回平均Dice系数得分
    # max(num_val_batches, 1)确保分母不为0，避免除零错误
    # 当验证集为空时，返回0
    return dice_score / max(num_val_batches, 1)
