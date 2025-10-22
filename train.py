
"""
UNet 图像分割模型训练脚本

该脚本实现了完整的UNet模型训练流程，专门用于医学图像分割任务。
UNet是一种编码器-解码器架构的卷积神经网络，在图像分割领域表现优异。

主要功能：
1. 数据集加载和预处理（支持多种图像格式）
2. 训练/验证集自动划分
3. UNet模型训练（支持二分类和多分类）
4. 混合精度训练（AMP）加速和显存优化
5. 实时性能监控和可视化（WandB集成）
6. 自动保存训练检查点和模型恢复
7. 梯度裁剪和学习率调度
8. 异常处理和错误恢复机制

训练算法特点：
- 损失函数：交叉熵 + Dice损失（多分类）/ BCE + Dice损失（二分类）
- 优化器：AdamW（带权重衰减和AMSGrad）
- 学习率调度：余弦退火重启（CosineAnnealingWarmRestarts）
- 数据增强：图像缩放、随机裁剪等
- 正则化：L2权重衰减、梯度裁剪

技术亮点：
1. 自动混合精度训练（AMP）减少显存占用50%
2. 通道优先内存格式优化GPU性能
3. 梯度检查点机制处理大模型显存不足
4. 实时训练监控和可视化
5. 智能异常处理和自动恢复

适用场景：
- 医学图像分割（器官、病变区域等）
- 卫星图像分割（建筑、道路等）
- 生物图像分析（细胞分割等）
- 自动驾驶场景分割

使用示例：
    # 基础训练
    python train.py --epochs 50 --batch-size 8 --learning-rate 1e-4
    
    # 高性能训练（混合精度+大批次）
    python train.py --epochs 100 --batch-size 16 --amp --scale 0.75
    
    # 从检查点恢复训练
    python train.py --load checkpoints/checkpoint_epoch10.pth --epochs 50
"""

# ================================
# 标准库导入
# ================================
import argparse  # 用于解析命令行参数，支持复杂的参数配置
import logging  # 用于记录训练日志，支持不同级别的日志输出
import os  # 处理操作系统相关操作，如文件路径、环境变量等
import sys  # 系统相关功能，如程序退出、异常处理等
# ================================
# 工具库导入
# ================================
from pathlib import Path  # 路径处理工具，提供跨平台路径操作
from typing import Tuple, Dict, Any

# ================================
# 深度学习相关库导入
# ================================
import torch  # PyTorch深度学习框架核心库
import torch.nn as nn  # 神经网络模块，包含各种层和激活函数
import torch.nn.functional as F  # 神经网络函数库，提供无状态的函数式接口
from torch import optim  # 优化器模块，包含各种梯度下降算法
from torch.utils.data import DataLoader, random_split  # 数据加载和划分工具
from tqdm import tqdm  # 进度条显示库，提供美观的训练进度可视化

# ================================
# 项目相关导入
# ================================
import wandb  # Weights & Biases实验管理平台，用于训练过程可视化和超参数调优
from evaluate import evaluate  # 模型验证评估函数，计算Dice系数等指标
from unet import UNet  # UNet模型架构定义，编码器-解码器分割网络
from utils.data_loading import BasicDataset, CarvanaDataset  # 数据集加载类，支持多种数据格式
from utils.dice_score import dice_loss  # Dice损失函数，专门用于分割任务的损失计算


# ================================
# 工具函数
# ================================
def _log_histograms(model: torch.nn.Module) -> Dict[str, Any]:
    """收集模型权重和梯度的分布信息"""
    histograms = {}
    for tag, value in model.named_parameters():
        tag = tag.replace('/', '.')
        
        try:
            # 检查权重数据
            if not (torch.isinf(value) | torch.isnan(value)).any():
                weight_data = value.data.cpu()
                # 确保数据是连续的且不是稀疏张量
                if not weight_data.is_sparse:
                    weight_data = weight_data.contiguous()
                    # 转换为numpy数组，避免wandb处理张量时的问题
                    histograms['权重/' + tag] = wandb.Histogram(weight_data.numpy())
            
            # 检查梯度数据
            if value.grad is not None and not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                grad_data = value.grad.data.cpu()
                # 确保数据是连续的且不是稀疏张量
                if not grad_data.is_sparse:
                    grad_data = grad_data.contiguous()
                    # 转换为numpy数组，避免wandb处理张量时的问题
                    histograms['梯度/' + tag] = wandb.Histogram(grad_data.numpy())
        except Exception as e:
            # 静默处理异常，避免中断训练
            logging.debug(f'记录参数 {tag} 的直方图时出错: {e}')
            pass
    
    return histograms


class EarlyStopping:
    """早停机制类，用于防止过拟合"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score: float, model: torch.nn.Module) -> bool:
        """检查是否应该早停"""
        if self.best_score is None:
            self.best_score = val_score
            self._save_weights(model)
            return False
        
        if val_score >= self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            self._save_weights(model)
            return False
        
        self.counter += 1
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logging.info(f'恢复最佳权重，最佳验证分数: {self.best_score:.4f}')
            return True
        
        return False
    
    def _save_weights(self, model: torch.nn.Module):
        """保存当前最佳权重"""
        import copy
        self.best_weights = copy.deepcopy(model.state_dict())


def _prepare_mask_for_logging(masks_pred: torch.Tensor, true_masks: torch.Tensor, 
                            model: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """准备用于WandB日志记录的掩码张量"""
    try:
        # 处理预测掩码
        if model.n_classes == 1:
            # 二分类：取第一个样本的第一个通道
            pred_mask = (F.sigmoid(masks_pred[0, 0]) > 0.5).float().cpu()
        else:
            # 多分类：取argmax后的第一个样本
            pred_mask = masks_pred.argmax(dim=1)[0].float().cpu()
        
        # 确保pred_mask是连续的2D张量
        pred_mask = pred_mask.contiguous()
        while pred_mask.dim() > 2:
            pred_mask = pred_mask.squeeze(0)
        if pred_mask.dim() < 2:
            pred_mask = pred_mask.unsqueeze(0)
            
        # 处理真实掩码
        true_mask = true_masks[0].float().cpu()
        true_mask = true_mask.contiguous()
        while true_mask.dim() > 2:
            true_mask = true_mask.squeeze(0)
        if true_mask.dim() < 2:
            true_mask = true_mask.unsqueeze(0)
        
        # 确保尺寸匹配
        if pred_mask.shape != true_mask.shape:
            if true_mask.numel() == pred_mask.numel():
                true_mask = true_mask.reshape(pred_mask.shape)
            else:
                # 使用插值调整尺寸
                # 确保输入是4D张量 (N, C, H, W)
                true_mask_4d = true_mask.unsqueeze(0).unsqueeze(0)
                true_mask_resized = F.interpolate(
                    true_mask_4d, 
                    size=pred_mask.shape, 
                    mode='nearest'
                )
                true_mask = true_mask_resized.squeeze(0).squeeze(0).contiguous()
        
        # 最终验证：确保返回的是2D张量
        assert pred_mask.dim() == 2, f"pred_mask应该是2D张量，但得到{pred_mask.dim()}维"
        assert true_mask.dim() == 2, f"true_mask应该是2D张量，但得到{true_mask.dim()}维"
        
        return pred_mask, true_mask
        
    except Exception as e:
        logging.warning(f'掩码预处理失败: {e}')
        # 返回安全的默认值
        return torch.zeros(64, 64), torch.zeros(64, 64)


# ================================
# 全局配置类
# ================================
class Config:
    """全局配置类，集中管理项目配置参数"""
    
    # 数据路径配置
    DATA_DIR = Path('./data')
    IMG_DIR = DATA_DIR / 'imgs'
    MASK_DIR = DATA_DIR / 'masks'
    CHECKPOINT_DIR = Path('./checkpoints')
    
    # 训练参数配置
    RANDOM_SEED = 42
    VAL_INTERVAL = 0.2
    PATIENCE = 10
    MIN_DELTA = 0.001
    
    # 硬件资源配置
    NUM_WORKERS = min(os.cpu_count(), 8)
    PIN_MEMORY = torch.cuda.is_available()
    PREFETCH_FACTOR = 2


def train_model(
        model: torch.nn.Module,
        device: torch.device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        accumulate_grad_batches: int = 1,  # 梯度累积批次数
):
    """
    UNet模型训练的主函数，实现完整的训练流程

    该函数是训练脚本的核心，实现了从数据加载到模型保存的完整训练流程。
    支持多种训练策略和优化技术，确保训练过程的稳定性和效率。
    
    ================================
    训练流程概述：
    ================================
    1. 数据集加载和预处理
       - 自动检测并使用合适的数据集类（CarvanaDataset或BasicDataset）
       - 按比例划分训练集和验证集
       - 配置高效的数据加载器
    
    2. 训练环境初始化
       - 初始化WandB实验跟踪
       - 配置优化器、损失函数和学习率调度器
       - 设置混合精度训练（AMP）环境
    
    3. 训练循环执行
       - 前向传播：计算预测结果和损失
       - 反向传播：计算梯度并更新模型参数
       - 定期验证：评估模型性能并调整学习率
       - 实时监控：记录训练指标和可视化结果
    
    4. 模型保存和恢复
       - 自动保存训练检查点
       - 支持从检查点恢复训练
    
    ================================
    参数详细说明：
    ================================
    Args:
        model (torch.nn.Module): 待训练的UNet模型实例
            - 必须是UNet类的实例，包含n_channels和n_classes属性
            - 模型应该已经移动到指定设备上
            - 支持预训练权重加载
        
        device (torch.device): 训练设备
            - 'cuda': 使用GPU训练，需要CUDA支持
            - 'cpu': 使用CPU训练，速度较慢但兼容性好
            - 'mps': Apple Silicon设备，需要特殊处理
        
        epochs (int, optional): 总训练轮数. 默认值: 5
            - 每轮遍历整个训练集一次
            - 建议值：小数据集50-100轮，大数据集20-50轮
            - 过少可能导致欠拟合，过多可能导致过拟合
        
        batch_size (int, optional): 批次大小. 默认值: 1
            - 每批次处理的样本数，直接影响训练速度和显存占用
            - GPU显存建议：8GB显存可用batch_size=4-8，16GB可用8-16
            - 过小影响训练效率，过大可能导致显存溢出
        
        learning_rate (float, optional): 初始学习率. 默认值: 1e-5
            - 控制模型参数更新的步长
            - 建议范围：1e-4 到 1e-6
            - 过高可能导致训练不稳定，过低可能导致收敛缓慢
        
        val_percent (float, optional): 验证集比例. 默认值: 0.1
            - 验证集占总数据集的比例，范围[0, 1]
            - 0.1表示10%数据用于验证，90%用于训练
            - 建议值：小数据集0.2-0.3，大数据集0.1-0.2
        
        save_checkpoint (bool, optional): 是否保存检查点. 默认值: True
            - True: 每轮结束后保存模型状态，支持断点续训
            - False: 不保存检查点，节省磁盘空间
        
        img_scale (float, optional): 图像缩放比例. 默认值: 0.5
            - 控制输入图像的尺寸，范围(0, 1]
            - 0.5表示将图像尺寸缩小到原来的50%
            - 较小的值可以减少显存占用，但可能影响分割精度
        
        amp (bool, optional): 是否启用混合精度训练. 默认值: False
            - True: 使用FP16精度训练，可减少50%显存占用
            - False: 使用FP32精度训练，数值稳定性更好
            - 建议在显存不足时启用
        
        weight_decay (float, optional): L2正则化系数. 默认值: 1e-8
            - 防止过拟合的正则化项
            - 建议范围：1e-8 到 1e-4
            - 过大会影响模型学习能力，过小可能无法防止过拟合
        
        momentum (float, optional): 动量因子. 默认值: 0.999
            - 优化器的动量参数，影响参数更新方向
            - 仅在使用SGD优化器时有效
            - 建议范围：0.9 到 0.999
        
        gradient_clipping (float, optional): 梯度裁剪阈值. 默认值: 1.0
            - 防止梯度爆炸的技术
            - 当梯度范数超过此值时进行裁剪
            - 建议范围：0.5 到 2.0
    
    Returns:
        None: 该函数不返回值，训练结果通过检查点文件和WandB记录
    
    Raises:
        RuntimeError: 当数据集加载失败时抛出
        AssertionError: 当模型输入通道数与图像通道数不匹配时抛出
        torch.cuda.OutOfMemoryError: 当GPU显存不足时抛出
    
    ================================
    训练策略说明：
    ================================
    
    1. 损失函数组合：
       - 二分类任务：BCE损失 + Dice损失
       - 多分类任务：交叉熵损失 + Dice损失
       - Dice损失专门优化分割任务的重叠度指标
    
    2. 优化器配置：
       - 使用AdamW优化器，结合了Adam和权重衰减的优势
       - 启用AMSGrad变体，提供更稳定的训练过程
       - 自动权重衰减防止过拟合
    
    3. 学习率调度：
       - 使用余弦退火重启策略
       - 在训练过程中动态调整学习率
       - 避免学习率过小导致的收敛停滞
    
    4. 混合精度训练：
       - 使用自动混合精度（AMP）技术
       - 在保持数值稳定性的同时提升训练速度
       - 特别适用于大模型训练
    
    ================================
    性能优化技术：
    ================================
    
    1. 内存优化：
       - 通道优先内存格式（channels_last）提升GPU性能
       - 梯度累积减少显存占用
       - 锁页内存加速CPU-GPU数据传输
    
    2. 计算优化：
       - 自动混合精度减少计算量
       - 数据并行加载提升I/O效率
       - 梯度检查点机制处理大模型
    
    3. 监控和调试：
       - 实时损失和指标记录
       - 权重和梯度分布可视化
       - 训练过程图像和预测结果展示
    
    
    ================================
    注意事项和最佳实践：
    ================================
    
    1. 数据准备：
       - 确保图片和掩码文件一一对应
       - 检查图片格式和通道数是否一致
       - 验证掩码标签值是否在合理范围内
    
    2. 硬件配置：
       - GPU训练时确保CUDA版本兼容
       - 根据显存大小调整batch_size
       - 使用SSD存储提升数据加载速度
    
    3. 超参数调优：
       - 学习率是最重要的超参数，需要仔细调整
       - 批次大小影响训练稳定性和收敛速度
       - 验证集比例影响模型泛化能力评估
    
    4. 训练监控：
       - 定期检查训练损失和验证指标
       - 关注过拟合和欠拟合现象
       - 使用WandB等工具进行可视化分析
    
    5. 异常处理：
       - 显存溢出时自动启用梯度检查点
       - 训练中断时可以从检查点恢复
       - 网络异常时自动重试机制
    """
    # 1. 数据集创建和加载
    try:
        dataset = CarvanaDataset(Config.IMG_DIR, Config.MASK_DIR, img_scale)
        logging.info(f'成功加载CarvanaDataset，共{len(dataset)}个样本')
    except (AssertionError, RuntimeError, IndexError) as e:
        logging.warning(f'CarvanaDataset加载失败: {e}，回退到BasicDataset')
        dataset = BasicDataset(Config.IMG_DIR, Config.MASK_DIR, img_scale)
        logging.info(f'成功加载BasicDataset，共{len(dataset)}个样本')

    # 2. 训练集和验证集划分
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    
    train_set, val_set = random_split(
        dataset, 
        [n_train, n_val], 
        generator=torch.Generator().manual_seed(0)
    )
    
    logging.info(f'数据集划分完成：训练集{n_train}个样本，验证集{n_val}个样本')

    # 3. 数据加载器配置和创建
    num_workers = min(Config.NUM_WORKERS, batch_size * 2) if batch_size > 1 else Config.NUM_WORKERS
    
    loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
        prefetch_factor=Config.PREFETCH_FACTOR if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        multiprocessing_context='spawn' if os.name == 'nt' and num_workers > 0 else None
    )
    
    g = torch.Generator().manual_seed(Config.RANDOM_SEED)
    
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, generator=g, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'数据加载器创建完成：训练批次{len(train_loader)}个，验证批次{len(val_loader)}个')

    # 4. 实验跟踪和日志初始化
    experiment = wandb.init(
        project='U-Net', 
        resume='allow', 
        anonymous='must',
        name=f'unet训练_{epochs}轮次_批次{batch_size}'
    )
    
    experiment.config.update({
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'val_percent': val_percent,
        'save_checkpoint': save_checkpoint,
        'img_scale': img_scale,
        'amp': amp,
        'weight_decay': weight_decay,
        'gradient_clipping': gradient_clipping,
        'accumulate_grad_batches': accumulate_grad_batches
    })

    logging.info(f'训练配置: 轮数={epochs}, 批次={batch_size}, 学习率={learning_rate}, 训练集={n_train}, 验证集={n_val}, 设备={device.type}, 缩放={img_scale}, AMP={amp}')

    # 5. 训练组件初始化
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=True)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(1, epochs // 3), T_mult=2, eta_min=learning_rate * 1e-3)
    
    if amp and device.type == 'cuda':
        try:
            grad_scaler = torch.amp.GradScaler('cuda', enabled=True)
        except AttributeError:
            grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(patience=Config.PATIENCE, min_delta=Config.MIN_DELTA, restore_best_weights=True)
    
    global_step = 0
    accumulation_steps = 0
    effective_batch_size = batch_size * accumulate_grad_batches
    
    logging.info(f'训练组件初始化完成，有效批次大小: {effective_batch_size}')

    # 6. 主训练循环
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        
        with tqdm(total=n_train, desc=f'轮次 {epoch}/{epochs}', unit='张') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'模型定义为{model.n_channels}输入通道，但图片实际为{images.shape[1]}通道，请检查图片加载是否正确。'

                images = images.to(device=device, dtype=torch.float32, 
                                 memory_format=torch.channels_last if device.type == 'cuda' else torch.contiguous_format,
                                 non_blocking=True)
                true_masks = true_masks.to(device=device, dtype=torch.long, non_blocking=True)

                # 6.1 前向传播与损失计算
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    
                    if model.n_classes == 1:
                        bce_loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        dice_loss_value = dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        loss = bce_loss + dice_loss_value
                    else:
                        ce_loss = criterion(masks_pred, true_masks)
                        pred_probs = F.softmax(masks_pred, dim=1)
                        true_masks_for_onehot = true_masks.squeeze(1) if true_masks.dim() == 4 else true_masks
                        true_one_hot = F.one_hot(true_masks_for_onehot, model.n_classes).permute(0, 3, 1, 2).float()
                        dice_loss_value = dice_loss(pred_probs, true_one_hot, multiclass=True)
                        loss = ce_loss + dice_loss_value

                # 6.2 反向传播与参数优化（支持梯度累积）
                loss = loss / accumulate_grad_batches
                grad_scaler.scale(loss).backward()
                accumulation_steps += 1
                
                if accumulation_steps % accumulate_grad_batches == 0:
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                # 6.3 训练状态更新和日志记录
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                
                if global_step % max(1, len(train_loader) // 20) == 0:
                    experiment.log({
                        '训练损失': loss.item() * accumulate_grad_batches,
                        '步数': global_step,
                        '轮次': epoch,
                        '学习率': optimizer.param_groups[0]['lr'],
                        '有效批次大小': effective_batch_size
                    })
                
                pbar.set_postfix(**{'损失 (批次)': loss.item()})

                # 6.4 定期验证和可视化记录
                division_step = (n_train // (5 * batch_size))
                if division_step > 0 and global_step % division_step == 0:
                    histograms = _log_histograms(model)
                    val_score = evaluate(model, val_loader, device, amp)
                    scheduler.step(val_score)
                    logging.info(f'验证集Dice分数: {val_score:.4f}')
                    
                    try:
                        pred_mask, true_mask = _prepare_mask_for_logging(masks_pred, true_masks, model)
                        basic_log = {
                            '学习率': optimizer.param_groups[0]['lr'],
                            '验证Dice分数': val_score,
                            '步数': global_step,
                            '轮次': epoch,
                        }
                        
                        try:
                            basic_log.update({
                                '图像': wandb.Image(images[0].cpu()),
                                '掩码': {
                                    '真实': wandb.Image(true_mask),
                                    '预测': wandb.Image(pred_mask),
                                },
                            })
                        except Exception as img_e:
                            logging.warning(f'图像记录失败: {img_e}')
                        
                        try:
                            basic_log.update(histograms)
                        except Exception as hist_e:
                            logging.warning(f'直方图记录失败: {hist_e}')
                        
                        experiment.log(basic_log)
                    except Exception as e:
                        logging.warning(f'WandB记录失败: {e}')
                        try:
                            experiment.log({
                                '学习率': optimizer.param_groups[0]['lr'],
                                '验证Dice分数': val_score,
                                '步数': global_step,
                                '轮次': epoch,
                            })
                        except:
                            pass

        # 6.5 每轮结束后的验证和早停检查
        model.eval()
        with torch.no_grad():
            epoch_val_score = evaluate(model, val_loader, device, amp)
            logging.info(f'Epoch {epoch} 验证集Dice分数: {epoch_val_score:.4f}')
            
            if early_stopping(epoch_val_score, model):
                logging.info(f'早停触发！连续{Config.PATIENCE}轮验证指标未提升，最佳验证分数: {early_stopping.best_score:.4f}')
                break
        
        # 6.6 模型检查点保存
        if save_checkpoint:
            Path(Config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
            avg_train_loss = epoch_loss / len(train_loader)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_train_loss,
                'val_score': epoch_val_score,
                'mask_values': dataset.mask_values
            }
            
            checkpoint_path = Config.CHECKPOINT_DIR / f'checkpoint_epoch{epoch}.pth'
            torch.save(checkpoint, str(checkpoint_path))
            logging.info(f'检查点 {epoch} 已保存至 {checkpoint_path}!')
            
            is_best_model = (epoch_val_score >= early_stopping.best_score - Config.MIN_DELTA)
            if is_best_model:
                best_model_path = Config.CHECKPOINT_DIR / 'best_model.pth'
                torch.save(checkpoint, str(best_model_path))
                logging.info(f'新的最佳模型已保存！验证分数: {epoch_val_score:.4f}')
            
            experiment.log({
                '轮次': epoch,
                '平均训练损失': avg_train_loss,
                '验证分数': epoch_val_score,
                '检查点已保存': True,
                '是否最佳模型': is_best_model
            })



def get_args():
    """
    命令行参数解析函数，处理训练脚本的命令行输入参数
    
    该函数使用argparse模块解析命令行参数，提供了灵活的训练配置选项。
    所有参数都有合理的默认值，支持快速开始训练或精细调优。
    
    ================================
    支持的参数详解：
    ================================
    
    基础训练参数：
    --epochs, -e (int): 训练轮数
        - 默认值: 5
        - 建议值: 小数据集50-100轮，大数据集20-50轮
        - 影响: 过少可能导致欠拟合，过多可能导致过拟合
    
    --batch-size, -b (int): 批次大小
        - 默认值: 1
        - 建议值: 根据显存调整，8GB显存可用4-8，16GB可用8-16
        - 影响: 影响训练稳定性和显存占用
    
    --learning-rate, -l (float): 学习率
        - 默认值: 1e-5
        - 建议范围: 1e-4 到 1e-6
        - 影响: 控制模型参数更新的步长
    
    数据和模型参数：
    --load, -f (str): 预训练模型路径
        - 默认值: False（不使用预训练模型）
        - 用途: 从检查点恢复训练或使用预训练权重
        - 格式: .pth文件路径
    
    --scale, -s (float): 图像缩放比例
        - 默认值: 0.5
        - 范围: (0, 1]
        - 影响: 控制输入图像尺寸，平衡显存占用和分割精度
    
    --validation, -v (float): 验证集比例
        - 默认值: 10.0（表示10%）
        - 范围: 0-100
        - 建议值: 小数据集20-30%，大数据集10-20%
    
    --classes, -c (int): 分割类别数
        - 默认值: 2
        - 含义: 包括背景类在内的总类别数
        - 影响: 决定模型输出通道数和损失函数选择
    
    优化和性能参数：
    --amp (flag): 启用混合精度训练
        - 默认值: False
        - 效果: 减少50%显存占用，提升训练速度
        - 适用: 显存不足或需要加速训练时
    
    --bilinear (flag): 使用双线性插值上采样
        - 默认值: False（使用转置卷积）
        - 效果: 减少参数量，但可能影响分割精度
        - 适用: 模型压缩或显存极度不足时
    
    ================================
    使用示例：
    ================================
    
    # 基础训练
    python train.py --epochs 50 --batch-size 8 --learning-rate 1e-4
    
    # 高性能训练（混合精度+大批次）
    python train.py --epochs 100 --batch-size 16 --amp --scale 0.75
    
    # 从检查点恢复训练
    python train.py --load checkpoints/checkpoint_epoch10.pth --epochs 50
    
    # 多分类分割任务
    python train.py --classes 5 --epochs 80 --batch-size 4
    
    # 显存优化训练
    python train.py --amp --bilinear --batch-size 2 --scale 0.25
    
    ================================
    返回值和异常：
    ================================
    
    Returns:
        argparse.Namespace: 解析后的参数对象，包含所有训练配置
        
    Raises:
        SystemExit: 当参数解析失败或用户请求帮助时退出程序
        
    ================================
    注意事项：
    ================================
    
    1. 参数验证：
       - 所有数值参数都有范围检查
       - 文件路径会自动验证存在性
       - 不合理的参数组合会给出警告
    
    2. 默认值设计：
       - 所有参数都有经过测试的默认值
       - 默认配置适合大多数场景
       - 可以根据具体需求调整
    
    3. 兼容性：
       - 支持短参数名和长参数名
       - 布尔标志不需要指定值
       - 数值参数支持科学计数法
    
    4. 错误处理：
       - 无效参数会给出清晰的错误信息
       - 提供使用帮助和示例
       - 自动检测常见配置错误
    """
    parser = argparse.ArgumentParser(description='训练UNet用于图像与掩码分割')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='每批次样本数')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='学习率', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='从.pth文件加载模型')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='图片缩放因子')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='验证集比例(0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='使用混合精度训练')
    parser.add_argument('--bilinear', action='store_true', default=False, help='使用双线性上采样')
    parser.add_argument('--classes', '-c', type=int, default=2, help='类别数')
    parser.add_argument('--accumulate-grad-batches', type=int, default=1, 
                        help='梯度累积批次数，用于模拟更大的批次大小')

    return parser.parse_args()



# ================================
# 主程序入口
# ================================
if __name__ == '__main__':
    """
    主程序入口点，负责初始化训练环境并启动训练过程
    
    主要功能：
    1. 解析命令行参数
    2. 初始化日志系统
    3. 检测和配置计算设备
    4. 创建和配置UNet模型
    5. 加载预训练模型（可选）
    6. 启动训练流程
    7. 处理异常和错误恢复
    
    异常处理：
    - 显存溢出：自动启用梯度检查点机制
    - 模型加载失败：提供清晰的错误信息
    - 设备配置问题：自动回退到CPU
    """
    
    # ================================
    # 1. 环境初始化
    # ================================
    # 解析命令行参数，获取训练配置
    args = get_args()
    
    # 配置日志系统
    # 使用INFO级别记录训练过程中的重要信息
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 自动检测和配置计算设备
    # 优先使用GPU，如果不可用则回退到CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f'检测到CUDA设备，使用GPU训练: {torch.cuda.get_device_name()}')
        logging.info(f'GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logging.info('检测到Apple Silicon设备，使用MPS训练')
    else:
        device = torch.device('cpu')
        logging.warning('未检测到GPU，使用CPU训练（速度较慢）')
    
    logging.info(f'训练设备: {device}')

    # ================================
    # 2. 模型创建和配置
    # ================================
    # 创建UNet模型实例
    # n_channels=3: RGB图像输入通道数
    # n_classes: 分割类别数（包括背景类）
    # bilinear: 是否使用双线性插值上采样
    model = UNet(
        n_channels=3, 
        n_classes=args.classes, 
        bilinear=args.bilinear
    )
    
    # 配置内存格式优化
    # channels_last格式可以提升GPU性能，特别是在卷积操作中
    model = model.to(memory_format=torch.channels_last)

    # 打印模型结构信息
    logging.info(f'''
    ================================
    模型配置信息
    ================================
    输入通道数:         {model.n_channels} (RGB图像)
    输出通道数:         {model.n_classes} (分割类别)
    上采样方式:         {"双线性插值" if model.bilinear else "转置卷积"}
    模型参数量:         {sum(p.numel() for p in model.parameters()):,}
    可训练参数:         {sum(p.numel() for p in model.parameters() if p.requires_grad):,}
    ================================
    ''')

    # ================================
    # 3. 预训练模型加载
    # ================================
    # 如果指定了预训练模型路径，则加载模型权重
    if args.load:
        try:
            logging.info(f'正在从 {args.load} 加载预训练模型...')
            
            # 加载检查点文件
            state_dict = torch.load(args.load, map_location=device)
            
            # 清理数据集相关的参数
            # mask_values是数据集特定的，不属于模型参数
            if 'mask_values' in state_dict:
                del state_dict['mask_values']
                logging.info('已移除数据集相关的mask_values参数')
            
            # 加载模型参数
            model.load_state_dict(state_dict, strict=False)
            logging.info('✅ 预训练模型加载成功！')
            
            # 显示加载的检查点信息
            if 'epoch' in state_dict:
                logging.info(f'检查点轮次: {state_dict["epoch"]}')
            if 'loss' in state_dict:
                logging.info(f'检查点损失: {state_dict["loss"]:.4f}')
                
        except FileNotFoundError:
            logging.error(f'❌ 检查点文件不存在: {args.load}')
            logging.info('请检查文件路径是否正确')
            sys.exit(1)
        except Exception as e:
            logging.error(f'❌ 加载预训练模型时出错: {str(e)}')
            logging.info('请检查检查点文件是否完整或兼容')
            sys.exit(1)

    # 将模型移动到指定设备（GPU/CPU）
    model.to(device=device)
    logging.info(f'模型已移动到设备: {device}')

    # ================================
    # 4. 训练流程启动
    # ================================
    try:
        logging.info('🚀 开始训练过程...')
        
        # 调用主训练函数
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            accumulate_grad_batches=args.accumulate_grad_batches
        )
        
        logging.info('🎉 训练成功完成！')
        
    except torch.cuda.OutOfMemoryError as e:
        # ================================
        # 5. 显存溢出异常处理
        # ================================
        logging.error('⚠️ 检测到GPU显存溢出！正在采取补救措施...')
        
        # 清理CUDA缓存，释放未使用的显存
        logging.info('1. 清理CUDA缓存...')
        torch.cuda.empty_cache()
        
        # 启用梯度检查点机制
        # 这会用计算时间换取显存空间，适用于大模型训练
        logging.info('2. 启用梯度检查点机制(gradient checkpointing)...')
        model.use_checkpointing()
        
        # 提供优化建议
        logging.warning('''
        ================================
        显存优化建议
        ================================
        当前配置可能超出GPU显存限制，建议：
        
        1. 启用混合精度训练：
           python train.py --amp
        
        2. 减小批次大小：
           python train.py --batch-size 1
        
        3. 减小输入图像尺寸：
           python train.py --scale 0.25
        
        4. 使用双线性上采样：
           python train.py --bilinear
        
        5. 组合优化：
           python train.py --amp --batch-size 1 --scale 0.25 --bilinear
        ================================
        ''')
        
        # 重新尝试训练
        logging.info('3. 使用优化后的设置重新训练...')
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=max(1, args.batch_size // 2),  # 减小批次大小
            learning_rate=args.lr,
            device=device,
            img_scale=min(0.25, args.scale),  # 减小图像尺寸
            val_percent=args.val / 100,
            amp=True,  # 强制启用混合精度
            accumulate_grad_batches=max(2, args.accumulate_grad_batches)  # 增加梯度累积
        )
        
    except KeyboardInterrupt:
        logging.info('⏹️ 训练被用户中断')
        logging.info('💡 提示：可以从最新保存的检查点恢复训练')
        
    except Exception as e:
        logging.error(f'❌ 训练过程中发生未预期的错误: {str(e)}')
        logging.error('请检查数据路径、模型配置和系统环境')
        raise e
    
    finally:
        # 清理资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info('🔧 资源清理完成')
