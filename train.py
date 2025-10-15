
"""
UNet 图像分割模型训练脚本
主要功能：
1. 加载和预处理图像数据
2. 训练UNet模型
3. 定期验证模型性能
4. 保存训练检查点
5. 可视化训练过程
"""

# 标准库导入
import argparse  # 用于解析命令行参数
import logging   # 用于记录训练日志
import os        # 处理操作系统相关操作
import random    # 随机数生成
import sys       # 系统相关功能

# 深度学习相关库导入
import torch            # PyTorch深度学习框架
import torch.nn as nn   # 神经网络模块
import torch.nn.functional as F  # 神经网络函数库
import torchvision.transforms as transforms        # 图像变换工具
import torchvision.transforms.functional as TF     # 图像变换函数
from torch import optim   # 优化器
from torch.utils.data import DataLoader, random_split  # 数据加载和划分工具

# 工具库导入
from pathlib import Path  # 路径处理工具
from tqdm import tqdm     # 进度条显示

# 项目相关导入
import wandb  # 用于实验管理和训练过程可视化
from evaluate import evaluate  # 模型验证评估函数
from unet import UNet         # UNet模型架构定义
from utils.data_loading import BasicDataset, CarvanaDataset  # 数据集加载类
from utils.dice_score import dice_loss  # Dice系数计算（分割任务评估指标）


# 定义全局配置类
class Config:
    """
    全局配置类，集中管理项目配置参数
    包含：
    1. 数据路径配置
    2. 训练参数配置
    3. 硬件资源配置
    """
    # 数据相关路径配置
    DATA_DIR = Path('./data')          # 数据根目录
    IMG_DIR = DATA_DIR / 'imgs'        # 训练图片存储路径
    MASK_DIR = DATA_DIR / 'masks'      # 标注掩码存储路径
    CHECKPOINT_DIR = Path('./checkpoints')  # 模型检查点保存路径
    
    # 训练相关参数配置
    RANDOM_SEED = 42        # 随机数种子，确保实验可重复性
    VAL_INTERVAL = 0.2     # 验证间隔（每训练多少轮进行一次验证）
    
    # 硬件资源相关配置
    NUM_WORKERS = min(os.cpu_count(), 8)  # 数据加载线程数，避免CPU过载
    PIN_MEMORY = torch.cuda.is_available()  # GPU训练时启用锁页内存，加速数据传输
    PREFETCH_FACTOR = 2  # 预加载批次数，平衡内存占用和加载速度


def train_model(
        model,
        device,
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
):
    """
    UNet模型训练的主函数，实现完整的训练流程

    功能：
    1. 数据集加载和划分
    2. 模型训练和验证
    3. 性能监控和可视化
    4. 模型保存和恢复

    参数说明：
        model: torch.nn.Module
            待训练的UNet模型实例
        
        device: torch.device
            训练设备，可以是'cuda'或'cpu'
        
        epochs: int, 默认=5
            总训练轮数，每轮遍历整个训练集一次
        
        batch_size: int, 默认=1
            每批次处理的样本数，根据显存大小调整
        
        learning_rate: float, 默认=1e-5
            初始学习率，影响模型收敛速度和稳定性
        
        val_percent: float, 默认=0.1
            验证集占总数据集的比例，范围[0,1]
        
        save_checkpoint: bool, 默认=True
            是否保存训练检查点，用于断点续训
        
        img_scale: float, 默认=0.5
            图像缩放比例，用于控制内存占用
        
        amp: bool, 默认=False
            是否启用自动混合精度训练，可降低显存占用
        
        weight_decay: float, 默认=1e-8
            L2正则化系数，用于防止过拟合
        
        momentum: float, 默认=0.999
            动量因子，影响优化器的更新行为
        
        gradient_clipping: float, 默认=1.0
            梯度裁剪阈值，防止梯度爆炸

    返回：
        None

    注意事项：
    1. 确保数据集格式正确（图片与掩码对应）
    2. 根据显存大小适当调整batch_size
    3. 如遇到显存溢出，可尝试开启amp
    4. 可通过wandb平台实时监控训练过程
    """
    # 1. Create dataset

    # 1. 创建数据集
    try:
        dataset = CarvanaDataset(Config.IMG_DIR, Config.MASK_DIR, img_scale)  # 优先使用Carvana数据集
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(Config.IMG_DIR, Config.MASK_DIR, img_scale)    # 失败则回退到BasicDataset

    # 2. Split into train / validation partitions

    # 2. 划分训练集和验证集
    n_val = int(len(dataset) * val_percent)  # 验证集样本数
    n_train = len(dataset) - n_val           # 训练集样本数
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))  # 固定随机种子

    # 3. 创建数据加载器，使用优化的参数
    loader_args = dict(
        batch_size=batch_size,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        prefetch_factor=Config.PREFETCH_FACTOR,
        persistent_workers=True  # 保持worker进程存活，减少重启开销
    )
    
    # 使用 generator 固定随机种子，确保可复现性
    g = torch.Generator()
    g.manual_seed(42)
    
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        drop_last=True,  # 丢弃不完整的最后一批，避免批次大小不一致
        **loader_args
    )
    
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        drop_last=True,
        **loader_args
    )

    # (Initialize logging)

    # 初始化wandb实验
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )


    # 打印训练参数信息
    logging.info(f'''开始训练:
        轮数:           {epochs}
        批大小:         {batch_size}
        学习率:         {learning_rate}
        训练集大小:     {n_train}
        验证集大小:     {n_val}
        是否保存检查点: {save_checkpoint}
        设备:           {device.type}
        图片缩放:       {img_scale}
        混合精度:       {amp}
    ''')

    # 4. 设置优化器、损失函数、学习率调度器和AMP缩放器
    optimizer = optim.AdamW(  # 使用AdamW优化器，更好的收敛性能
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        amsgrad=True  # 启用AMSGrad变体，提供更稳定的训练
    )
    
    # 使用CosineAnnealingWarmRestarts调度器，在训练后期提供更好的学习率调整
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=epochs // 3,  # 第一次重启的周期
        T_mult=2,  # 每次重启后周期翻倍
        eta_min=learning_rate * 1e-3)  # 最小学习率
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)  # 混合精度缩放器
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()  # 分类/二分类损失
    global_step = 0  # 全局步数计数

    # 5. Begin training

    # 5. 开始训练循环
    for epoch in range(1, epochs + 1):
        model.train()  # 设置为训练模式
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']  # 获取图片和掩码

                # 检查图片通道数是否与模型一致
                assert images.shape[1] == model.n_channels, \
                    f'模型定义为{model.n_channels}输入通道，但图片实际为{images.shape[1]}通道，请检查图片加载是否正确。'

                # 将数据转移到指定设备
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # 前向传播与损失计算（支持混合精度）
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        # 二分类：BCE损失 + Dice损失
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        # 多分类：交叉熵损失 + Dice损失
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                # 反向传播与优化
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()  # 梯度缩放反向传播
                grad_scaler.unscale_(optimizer)     # 取消缩放
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)  # 梯度裁剪
                grad_scaler.step(optimizer)         # 优化器更新
                grad_scaler.update()                # 缩放器更新

                pbar.update(images.shape[0])        # 更新进度条
                global_step += 1
                epoch_loss += loss.item()           # 累加损失
                experiment.log({                    # wandb记录训练损失
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # 每隔一定步数进行一次验证与记录
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            # 记录权重和梯度的分布直方图
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        # 验证集评估，更新学习率调度器
                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('验证集Dice分数: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        # 每轮训练结束后保存模型检查点
        if save_checkpoint:
            Path(Config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss / len(train_loader),
                'mask_values': dataset.mask_values
            }
            checkpoint_path = Config.CHECKPOINT_DIR / f'checkpoint_epoch{epoch}.pth'
            torch.save(checkpoint, str(checkpoint_path))
            logging.info(f'检查点 {epoch} 已保存至 {checkpoint_path}!')



def get_args():
    """
    命令行参数解析函数，处理训练脚本的命令行输入参数
    
    支持的参数：
    --epochs, -e        : 训练轮数
    --batch-size, -b    : 批次大小
    --learning-rate, -l : 学习率
    --load, -f         : 加载预训练模型路径
    --scale, -s        : 图像缩放比例
    --validation, -v    : 验证集比例
    --amp              : 是否使用混合精度训练
    --bilinear         : 是否使用双线性插值上采样
    --classes, -c      : 分割类别数
    
    使用示例：
    >>> python train.py --epochs 100 --batch-size 16 --learning-rate 1e-4 --amp
    
    返回：
        argparse.Namespace: 解析后的参数对象
    
    注意：
    1. 所有参数都有默认值，可以不指定
    2. amp和bilinear为布尔标志，不需要值
    3. 验证集比例为百分比形式(0-100)
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

    return parser.parse_args()



if __name__ == '__main__':
    # 主程序入口
    args = get_args()  # 解析命令行参数

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择GPU或CPU
    logging.info(f'使用设备 {device}')

    # 根据数据类型设置模型参数
    # n_channels=3 表示RGB图片
    # n_classes 表示每个像素的类别数
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'网络结构:\n'
                 f'\t{model.n_channels} 输入通道\n'
                 f'\t{model.n_classes} 输出通道(类别)\n'
                 f'\t{"双线性" if model.bilinear else "转置卷积"} 上采样')

    # 加载预训练模型（如果指定）
    if args.load:
        try:
            logging.info(f'正在从{args.load}加载预训练模型...')
            state_dict = torch.load(args.load, map_location=device)
            # 移除mask_values，因为这是数据集相关的，不属于模型参数
            if 'mask_values' in state_dict:
                del state_dict['mask_values']
            # 加载模型参数
            model.load_state_dict(state_dict)
            logging.info('预训练模型加载成功！')
        except Exception as e:
            logging.error(f'加载预训练模型时出错: {str(e)}')
            sys.exit(1)

    # 将模型移动到指定设备（GPU/CPU）
    model.to(device=device)

    # 开始训练流程
    try:
        logging.info('开始训练过程...')
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
        logging.info('训练成功完成！')
        
    except torch.cuda.OutOfMemoryError:
        # 处理显存溢出异常
        logging.error('⚠️ 检测到显存溢出！采取补救措施：')
        logging.info('1. 清理CUDA缓存')
        torch.cuda.empty_cache()
        
        logging.info('2. 启用梯度检查点机制(gradient checkpointing)')
        model.use_checkpointing()
        
        logging.info('3. 使用优化后的设置重新训练')
        logging.warning('注意：启用检查点会降低训练速度，建议：')
        logging.warning('- 使用 --amp 参数启用混合精度训练')
        logging.warning('- 减小batch_size')
        logging.warning('- 减小输入图像尺寸')
        
        # 重新尝试训练
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
