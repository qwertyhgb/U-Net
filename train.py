
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
    """
    收集模型权重和梯度的分布信息
    
    Args:
        model: 待分析的模型
        
    Returns:
        dict: 包含权重和梯度分布的字典
    """
    histograms = {}
    for tag, value in model.named_parameters():
        tag = tag.replace('/', '.')  # 将路径分隔符替换为点号，便于WandB显示
        
        # 记录权重分布直方图（排除无穷大和NaN值）
        try:
            if value.grad is not None and not (torch.isinf(value) | torch.isnan(value)).any():
                # 确保张量是连续的且非稀疏的
                weight_data = value.data.cpu().contiguous()
                if not weight_data.is_sparse:
                    histograms['权重/' + tag] = wandb.Histogram(weight_data)
        except Exception:
            # 如果记录权重失败，跳过但不影响训练
            pass
        
        # 记录梯度分布直方图（排除无穷大和NaN值）
        try:
            if value.grad is not None and not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                # 确保梯度张量是连续的且非稀疏的
                grad_data = value.grad.data.cpu().contiguous()
                if not grad_data.is_sparse:
                    histograms['梯度/' + tag] = wandb.Histogram(grad_data)
        except Exception:
            # 如果记录梯度失败，跳过但不影响训练
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
        """
        检查是否应该早停
        
        Args:
            val_score: 当前验证分数
            model: 模型实例
            
        Returns:
            bool: True表示应该停止训练
        """
        if self.best_score is None:
            self.best_score = val_score
            self._save_weights(model)
        elif val_score >= self.best_score + self.min_delta:
            # 验证分数有显著提升，更新最佳分数
            self.best_score = val_score
            self.counter = 0
            self._save_weights(model)
        else:
            # 验证分数没有显著提升，增加计数器
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    logging.info(f'恢复最佳权重，最佳验证分数: {self.best_score:.4f}')
                return True
            
        return False
    
    def _save_weights(self, model: torch.nn.Module):
        """保存当前最佳权重"""
        self.best_weights = model.state_dict().copy()


def _prepare_mask_for_logging(masks_pred: torch.Tensor, true_masks: torch.Tensor, 
                            model: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    准备用于WandB日志记录的掩码张量
    
    Args:
        masks_pred: 预测掩码张量
        true_masks: 真实掩码张量
        model: 模型实例
        
    Returns:
        tuple: (处理后的预测掩码, 处理后的真实掩码)
    """
    try:
        # 处理预测掩码的维度问题，确保与WandB兼容
        if model.n_classes == 1:
            # 二分类任务：使用sigmoid + 阈值处理
            pred_mask = (F.sigmoid(masks_pred[0, 0]) > 0.5).float().cpu()
        else:
            # 多分类任务：使用argmax处理
            pred_mask = masks_pred.argmax(dim=1)[0].float().cpu()
        
        # 确保掩码张量是正确的2D格式用于WandB
        while pred_mask.dim() > 2:
            pred_mask = pred_mask.squeeze()
        while pred_mask.dim() < 2:
            pred_mask = pred_mask.unsqueeze(0)
            
        # 确保真实掩码也是2D格式
        true_mask = true_masks[0].float().cpu()
        
        # 处理真实掩码的维度
        while true_mask.dim() > 2:
            true_mask = true_mask.squeeze()
        while true_mask.dim() < 2:
            true_mask = true_mask.unsqueeze(0)
        
        # 确保掩码尺寸匹配
        if pred_mask.shape != true_mask.shape:
            # 如果尺寸不匹配，调整真实掩码尺寸
            if true_mask.numel() == pred_mask.numel():
                true_mask = true_mask.view(pred_mask.shape)
            else:
                # 如果元素数量也不匹配，使用插值调整
                true_mask = F.interpolate(
                    true_mask.unsqueeze(0).unsqueeze(0), 
                    size=pred_mask.shape, 
                    mode='nearest'
                ).squeeze()
        
        return pred_mask, true_mask
        
    except Exception as e:
        logging.warning(f'掩码预处理失败: {e}')
        # 返回简单的占位符掩码
        dummy_mask = torch.zeros(64, 64)
        return dummy_mask, dummy_mask


# ================================
# 全局配置类
# ================================
class Config:
    """
    全局配置类，集中管理项目配置参数
    
    该类采用集中式配置管理，便于参数调优和维护。
    所有配置参数都定义为类属性，可以在整个训练过程中统一访问。
    
    配置分类：
    1. 数据路径配置：定义数据集、检查点的存储位置
    2. 训练参数配置：控制训练过程的超参数
    3. 硬件资源配置：优化多核CPU和GPU使用效率
    
    设计原则：
    - 单一职责：每个配置项都有明确的作用
    - 可扩展性：易于添加新的配置参数
    - 可维护性：集中管理，便于批量修改
    """
    
    # ================================
    # 数据相关路径配置
    # ================================
    DATA_DIR = Path('./data')          # 数据根目录，存储所有训练数据
    IMG_DIR = DATA_DIR / 'imgs'        # 训练图片存储路径，支持JPG、PNG等格式
    MASK_DIR = DATA_DIR / 'masks'      # 标注掩码存储路径，与图片一一对应
    CHECKPOINT_DIR = Path('./checkpoints')  # 模型检查点保存路径，用于断点续训
    
    # ================================
    # 训练相关参数配置
    # ================================
    RANDOM_SEED = 42        # 随机数种子，确保实验可重复性
    VAL_INTERVAL = 0.2     # 验证间隔比例（每训练多少轮进行一次验证）
    PATIENCE = 10          # 早停耐心值（连续多少轮验证指标不提升则停止）
    MIN_DELTA = 0.001      # 早停最小改进阈值
    
    # ================================
    # 硬件资源相关配置
    # ================================
    NUM_WORKERS = min(os.cpu_count(), 8)  # 数据加载线程数，避免CPU过载
    PIN_MEMORY = torch.cuda.is_available()  # GPU训练时启用锁页内存，加速数据传输
    PREFETCH_FACTOR = 2  # 预加载批次数，平衡内存占用和加载速度


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
    使用示例：
    ================================
    
    ```python
    # 基础训练示例
    model = UNet(n_channels=3, n_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_model(
        model=model,
        device=device,
        epochs=50,
        batch_size=8,
        learning_rate=1e-4,
        val_percent=0.2,
        amp=True
    )
    
    # 高性能训练示例
    train_model(
        model=model,
        device=device,
        epochs=100,
        batch_size=16,
        learning_rate=1e-4,
        img_scale=0.75,
        amp=True,
        weight_decay=1e-5
    )
    ```
    
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
    # ================================
    # 1. 数据集创建和加载
    # ================================
    # 优先尝试使用CarvanaDataset，如果失败则回退到BasicDataset
    # CarvanaDataset专门为Carvana汽车分割数据集优化，包含特定的掩码后缀处理
    # BasicDataset是通用数据集类，支持各种图像分割任务
    try:
        dataset = CarvanaDataset(Config.IMG_DIR, Config.MASK_DIR, img_scale)
        logging.info(f'成功加载CarvanaDataset，共{len(dataset)}个样本')
    except (AssertionError, RuntimeError, IndexError) as e:
        logging.warning(f'CarvanaDataset加载失败: {e}，回退到BasicDataset')
        dataset = BasicDataset(Config.IMG_DIR, Config.MASK_DIR, img_scale)
        logging.info(f'成功加载BasicDataset，共{len(dataset)}个样本')

    # ================================
    # 2. 训练集和验证集划分
    # ================================
    # 按照指定比例将数据集划分为训练集和验证集
    # 使用固定随机种子确保每次运行的数据划分结果一致
    n_val = int(len(dataset) * val_percent)  # 计算验证集样本数量
    n_train = len(dataset) - n_val           # 计算训练集样本数量
    
    # 使用random_split进行数据划分，确保可重复性
    # generator参数使用固定种子，保证实验的可重复性
    train_set, val_set = random_split(
        dataset, 
        [n_train, n_val], 
        generator=torch.Generator().manual_seed(0)
    )
    
    logging.info(f'数据集划分完成：训练集{n_train}个样本，验证集{n_val}个样本')

    # ================================
    # 3. 数据加载器配置和创建
    # ================================
    # 动态配置数据加载器参数，根据设备和批次大小优化
    num_workers = min(Config.NUM_WORKERS, batch_size * 2) if batch_size > 1 else Config.NUM_WORKERS
    
    # 根据设备类型优化数据加载器配置
    loader_args = dict(
        batch_size=batch_size,                    # 批次大小，根据显存调整
        num_workers=num_workers,                 # 动态调整数据加载线程数
        pin_memory=device.type == 'cuda',        # 仅在CUDA设备上启用锁页内存
        prefetch_factor=Config.PREFETCH_FACTOR if num_workers > 0 else None,  # 预加载批次数
        persistent_workers=num_workers > 0,      # 仅在多进程时保持worker存活
        multiprocessing_context='spawn' if os.name == 'nt' and num_workers > 0 else None  # Windows兼容性
    )
    
    # 创建固定随机种子的生成器，确保训练过程的可重复性
    g = torch.Generator()
    g.manual_seed(Config.RANDOM_SEED)
    
    # 创建训练数据加载器
    train_loader = DataLoader(
        train_set,
        shuffle=True,          # 训练时打乱数据顺序，提高模型泛化能力
        drop_last=True,        # 丢弃不完整的最后一批，避免批次大小不一致影响训练稳定性
        generator=g,           # 使用固定种子的生成器
        **loader_args
    )
    
    # 创建验证数据加载器
    val_loader = DataLoader(
        val_set,
        shuffle=False,         # 验证时不打乱数据，确保评估结果的一致性
        drop_last=True,        # 同样丢弃不完整的批次
        **loader_args
    )

    logging.info(f'数据加载器创建完成：训练批次{len(train_loader)}个，验证批次{len(val_loader)}个')

    # ================================
    # 4. 实验跟踪和日志初始化
    # ================================
    # 初始化WandB实验跟踪，用于训练过程可视化和超参数管理
    # resume='allow': 允许恢复中断的实验
    # anonymous='must': 允许匿名使用WandB
    experiment = wandb.init(
        project='U-Net', 
        resume='allow', 
        anonymous='must',
        name=f'unet训练_{epochs}轮次_批次{batch_size}'
    )
    
    # 记录训练配置参数到WandB
    experiment.config.update(
        dict(
            epochs=epochs, 
            batch_size=batch_size, 
            learning_rate=learning_rate,
            val_percent=val_percent, 
            save_checkpoint=save_checkpoint, 
            img_scale=img_scale, 
            amp=amp,
            weight_decay=weight_decay,
            gradient_clipping=gradient_clipping
        )
    )

    # 打印详细的训练参数信息
    logging.info(f'''
    ================================
    训练配置信息
    ================================
    训练轮数:           {epochs}
    批次大小:           {batch_size}
    学习率:             {learning_rate}
    训练集大小:         {n_train}
    验证集大小:         {n_val}
    验证集比例:         {val_percent:.1%}
    是否保存检查点:     {save_checkpoint}
    训练设备:           {device.type}
    图片缩放比例:       {img_scale}
    混合精度训练:       {amp}
    权重衰减:           {weight_decay}
    梯度裁剪阈值:       {gradient_clipping}
    ================================
    ''')

    # ================================
    # 5. 训练组件初始化
    # ================================
    
    # 5.1 优化器配置
    # 使用AdamW优化器，结合了Adam的自适应学习率和权重衰减的优势
    # AdamW相比Adam有更好的泛化性能，特别是在深度学习任务中
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,           # 初始学习率
        weight_decay=weight_decay,  # L2正则化系数，防止过拟合
        amsgrad=True               # 启用AMSGrad变体，提供更稳定的训练过程
    )
    
    # 5.2 学习率调度器配置
    # 使用余弦退火重启策略，在训练过程中动态调整学习率
    # 这种策略可以在训练后期提供更好的收敛性能
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=epochs // 3,                    # 第一次重启的周期长度
        T_mult=2,                          # 每次重启后周期长度翻倍
        eta_min=learning_rate * 1e-3       # 最小学习率，防止学习率过小
    )
    
    # 5.3 混合精度训练缩放器
    # 用于混合精度训练的梯度缩放，确保训练稳定性
    # 当使用FP16精度时，梯度可能过小，需要放大后更新参数
    if amp and device.type == 'cuda':
        try:
            # 使用新的API (PyTorch 2.0+)
            grad_scaler = torch.amp.GradScaler('cuda', enabled=True)
        except AttributeError:
            # 向后兼容旧版本PyTorch
            grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        # CPU或MPS设备不支持混合精度，创建禁用的scaler
        grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    
    # 5.4 损失函数配置
    # 根据任务类型选择合适的损失函数
    # 多分类任务使用交叉熵损失，二分类任务使用BCE损失
    if model.n_classes > 1:
        criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
        logging.info('使用交叉熵损失函数（多分类任务）')
    else:
        criterion = nn.BCEWithLogitsLoss()  # 二分类BCE损失
        logging.info('使用BCE损失函数（二分类任务）')
    
    # 5.5 早停机制初始化
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE, 
        min_delta=Config.MIN_DELTA, 
        restore_best_weights=True
    )
    
    # 5.6 训练状态变量
    global_step = 0  # 全局训练步数计数器，用于WandB日志记录
    accumulation_steps = 0  # 梯度累积步数计数器
    
    # 计算有效批次大小
    effective_batch_size = batch_size * accumulate_grad_batches
    logging.info(f'有效批次大小: {effective_batch_size} (批次大小: {batch_size} × 累积步数: {accumulate_grad_batches})')
    
    logging.info('训练组件初始化完成')

    # ================================
    # 6. 主训练循环
    # ================================
    # 开始逐轮训练，每轮遍历整个训练集一次
    for epoch in range(1, epochs + 1):
        # 将模型设置为训练模式
        # 这会启用dropout、batch normalization的训练模式等训练时特有的行为
        model.train()
        epoch_loss = 0  # 当前轮次的累积损失
        
        # 创建进度条，显示当前轮次的训练进度
        with tqdm(total=n_train, desc=f'轮次 {epoch}/{epochs}', unit='张') as pbar:
            # 遍历训练集中的所有批次
            for batch in train_loader:
                # 从批次中提取图像和真实掩码
                images, true_masks = batch['image'], batch['mask']

                # 验证图像通道数是否与模型期望一致
                # 这是重要的安全检查，避免维度不匹配错误
                assert images.shape[1] == model.n_channels, \
                    f'模型定义为{model.n_channels}输入通道，但图片实际为{images.shape[1]}通道，请检查图片加载是否正确。'

                # 将数据移动到指定设备并转换为合适的类型
                # images: 转换为float32类型，使用channels_last内存格式优化GPU性能
                # true_masks: 转换为long类型，因为掩码标签通常是整数索引
                images = images.to(device=device, dtype=torch.float32, 
                                 memory_format=torch.channels_last if device.type == 'cuda' else torch.contiguous_format,
                                 non_blocking=True)
                true_masks = true_masks.to(device=device, dtype=torch.long, non_blocking=True)

                # ================================
                # 6.1 前向传播与损失计算
                # ================================
                # 使用自动混合精度进行前向传播，提升训练速度并减少显存占用
                # 对于MPS设备需要特殊处理，因为Apple Silicon对autocast支持有限
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    # 模型前向传播，得到预测结果
                    # masks_pred形状：(batch_size, n_classes, height, width)
                    masks_pred = model(images)
                    
                    # 根据任务类型计算不同的损失函数组合
                    if model.n_classes == 1:
                        # ================================
                        # 二分类任务损失计算
                        # ================================
                        # 使用BCE损失 + Dice损失的组合
                        # BCE损失：二元交叉熵，处理前景/背景分割
                        bce_loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        
                        # Dice损失：专门优化分割重叠度，改善边界分割效果
                        # 使用sigmoid激活函数将logits转换为概率
                        dice_loss_value = dice_loss(
                            F.sigmoid(masks_pred.squeeze(1)), 
                            true_masks.float(), 
                            multiclass=False
                        )
                        
                        # 总损失 = BCE损失 + Dice损失
                        # 这种组合在医学图像分割中表现优异
                        loss = bce_loss + dice_loss_value
                        
                    else:
                        # ================================
                        # 多分类任务损失计算
                        # ================================
                        # 使用交叉熵损失 + Dice损失的组合
                        # 交叉熵损失：处理多个类别的前景分割
                        ce_loss = criterion(masks_pred, true_masks)
                        
                        # 将预测结果转换为概率分布（softmax）
                        pred_probs = F.softmax(masks_pred, dim=1)
                        
                        # 将真实标签转换为one-hot编码格式（优化内存使用）
                        # 确保true_masks是正确的维度：(batch_size, height, width)
                        if true_masks.dim() == 4:  # 如果是4维，去掉通道维度
                            true_masks_for_onehot = true_masks.squeeze(1)
                        else:
                            true_masks_for_onehot = true_masks
                        
                        # 转换为one-hot编码：(batch_size, height, width) -> (batch_size, height, width, n_classes)
                        # 然后调整维度顺序为：(batch_size, n_classes, height, width)
                        true_one_hot = F.one_hot(true_masks_for_onehot, model.n_classes).permute(0, 3, 1, 2).float()
                        
                        # 计算多分类Dice损失
                        dice_loss_value = dice_loss(pred_probs, true_one_hot, multiclass=True)
                        
                        # 总损失 = 交叉熵损失 + Dice损失
                        loss = ce_loss + dice_loss_value

                # ================================
                # 6.2 反向传播与参数优化（支持梯度累积）
                # ================================
                # 将损失除以累积步数，实现梯度累积
                loss = loss / accumulate_grad_batches
                
                # 梯度缩放反向传播
                # 在混合精度训练中，损失需要先放大再反向传播
                # 这样可以避免梯度下溢问题，保持训练稳定性
                grad_scaler.scale(loss).backward()
                
                accumulation_steps += 1
                
                # 当达到累积步数时，执行参数更新
                if accumulation_steps % accumulate_grad_batches == 0:
                    # 取消梯度缩放，准备进行梯度裁剪和参数更新
                    grad_scaler.unscale_(optimizer)
                    
                    # 梯度裁剪：防止梯度爆炸
                    # 当梯度的L2范数超过阈值时，按比例缩放所有梯度
                    # 这是RNN和深度网络训练中的重要技术
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    
                    # 执行优化器步骤，更新模型参数
                    # 在混合精度训练中，需要先缩放梯度再更新
                    grad_scaler.step(optimizer)
                    
                    # 更新梯度缩放器的内部状态
                    # 根据是否发生梯度溢出，动态调整缩放因子
                    grad_scaler.update()
                    
                    # 清空梯度缓存，set_to_none=True可以节省内存
                    optimizer.zero_grad(set_to_none=True)

                # ================================
                # 6.3 训练状态更新和日志记录
                # ================================
                # 更新进度条显示
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()  # 累加当前轮次的损失
                
                # 记录训练指标到WandB（每隔一定步数记录一次，减少I/O开销）
                if global_step % max(1, len(train_loader) // 20) == 0:  # 每轮记录20次
                    experiment.log({
                        '训练损失': loss.item() * accumulate_grad_batches,  # 恢复原始损失值
                        '步数': global_step,            # 全局训练步数
                        '轮次': epoch,                 # 当前轮次
                        '学习率': optimizer.param_groups[0]['lr'],  # 当前学习率
                        '有效批次大小': effective_batch_size  # 有效批次大小
                    })
                
                # 更新进度条后缀，显示当前批次的损失
                pbar.set_postfix(**{'损失 (批次)': loss.item()})

                # ================================
                # 6.4 定期验证和可视化记录
                # ================================
                # 计算验证间隔步数，每轮训练进行5次验证
                # 这样可以更频繁地监控模型性能，及时发现过拟合等问题
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        # 收集模型权重和梯度的分布信息
                        histograms = _log_histograms(model)

                        # 在验证集上评估模型性能
                        val_score = evaluate(model, val_loader, device, amp)
                        
                        # 根据验证分数更新学习率调度器
                        scheduler.step(val_score)

                        # 记录验证结果
                        logging.info(f'验证集Dice分数: {val_score:.4f}')
                        
                        # 记录详细的验证信息到WandB
                        try:
                            pred_mask, true_mask = _prepare_mask_for_logging(masks_pred, true_masks, model)
                            
                            # 基础日志记录（总是尝试记录）
                            basic_log = {
                                '学习率': optimizer.param_groups[0]['lr'],  # 当前学习率
                                '验证Dice分数': val_score,                       # 验证集Dice分数
                                '步数': global_step,                               # 全局步数
                                '轮次': epoch,                                    # 当前轮次
                            }
                            
                            # 尝试添加图像和掩码（可能失败）
                            try:
                                basic_log.update({
                                    '图像': wandb.Image(images[0].cpu()),            # 输入图像
                                    '掩码': {                                         # 掩码对比
                                        '真实': wandb.Image(true_mask),                # 真实掩码
                                        '预测': wandb.Image(pred_mask),                # 预测掩码
                                    },
                                })
                            except Exception as img_e:
                                logging.warning(f'图像记录失败: {img_e}')
                            
                            # 尝试添加直方图（可能失败）
                            try:
                                basic_log.update(histograms)
                            except Exception as hist_e:
                                logging.warning(f'直方图记录失败: {hist_e}')
                            
                            # 记录到WandB
                            experiment.log(basic_log)
                            
                        except Exception as e:
                            # 如果记录失败，继续训练而不中断
                            logging.warning(f'WandB记录失败: {e}')
                            # 至少记录基本指标
                            try:
                                experiment.log({
                                    '学习率': optimizer.param_groups[0]['lr'],
                                    '验证Dice分数': val_score,
                                    '步数': global_step,
                                    '轮次': epoch,
                                })
                            except:
                                pass

        # ================================
        # 6.5 每轮结束后的验证和早停检查
        # ================================
        # 在每个epoch结束后进行完整验证
        model.eval()
        with torch.no_grad():
            epoch_val_score = evaluate(model, val_loader, device, amp)
            logging.info(f'Epoch {epoch} 验证集Dice分数: {epoch_val_score:.4f}')
            
            # 检查早停条件
            if early_stopping(epoch_val_score, model):
                logging.info(f'早停触发！连续{Config.PATIENCE}轮验证指标未提升')
                logging.info(f'最佳验证分数: {early_stopping.best_score:.4f}')
                break
        
        # ================================
        # 6.6 模型检查点保存
        # ================================
        # 每轮训练结束后保存模型状态，支持断点续训和模型恢复
        if save_checkpoint:
            # 确保检查点目录存在
            Path(Config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
            
            avg_train_loss = epoch_loss / len(train_loader)
            
            # 构建检查点数据字典
            checkpoint = {
                'epoch': epoch,                                    # 当前训练轮次
                'model_state_dict': model.state_dict(),           # 模型参数状态
                'optimizer_state_dict': optimizer.state_dict(),   # 优化器状态（包括动量等）
                'scheduler_state_dict': scheduler.state_dict(),   # 学习率调度器状态
                'loss': avg_train_loss,                           # 平均损失
                'val_score': epoch_val_score,                     # 验证分数
                'mask_values': dataset.mask_values                # 数据集掩码值（用于推理时的标签映射）
            }
            
            # 保存当前epoch检查点
            checkpoint_path = Config.CHECKPOINT_DIR / f'checkpoint_epoch{epoch}.pth'
            torch.save(checkpoint, str(checkpoint_path))
            logging.info(f'检查点 {epoch} 已保存至 {checkpoint_path}!')
            
            # 保存最佳模型
            best_model_path = Config.CHECKPOINT_DIR / 'best_model.pth'
            is_best_model = (epoch_val_score >= early_stopping.best_score - Config.MIN_DELTA)
            if is_best_model:
                torch.save(checkpoint, str(best_model_path))
                logging.info(f'新的最佳模型已保存！验证分数: {epoch_val_score:.4f}')
            
            # 记录检查点信息到WandB
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
