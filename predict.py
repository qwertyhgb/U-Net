"""
UNet模型预测脚本
用于加载训练好的UNet模型，对输入图像进行语义分割预测

运行示例:
    基本用法（单张图像）:
        python predict.py -i image.jpg -m checkpoints/model.pth
    
    批量预测多张图像:
        python predict.py -i img1.jpg img2.jpg img3.jpg -m model.pth
    
    指定输出文件名:
        python predict.py -i input.jpg -o output.png -m model.pth
    
    输出到指定文件夹:
        python predict.py -i image.jpg -m model.pth --output-dir ./results
    
    批量输出到文件夹（保持原文件名）:
        python predict.py -i img1.jpg img2.jpg -m model.pth --output-dir ./masks
    
    调整缩放因子和阈值:
        python predict.py -i image.jpg -m model.pth -s 1.0 -t 0.6
    
    启用可视化:
        python predict.py -i image.jpg -m model.pth -v
    
    多分类任务（3个类别）:
        python predict.py -i image.jpg -m model.pth -c 3
    
    使用双线性上采样:
        python predict.py -i image.jpg -m model.pth --bilinear
    
    只可视化不保存:
        python predict.py -i image.jpg -m model.pth -v -n

参数说明:
    -i, --input: 输入图像路径（必需），支持多个文件
    -m, --model: 模型文件路径（默认: MODEL.pth）
    -o, --output: 输出图像路径（可选），未指定时自动生成
    --output-dir: 输出文件夹路径（可选），所有掩码将保存到此文件夹
    -s, --scale: 图像缩放因子（默认: 0.5），范围0-1
    -t, --mask-threshold: 二分类阈值（默认: 0.5），范围0-1
    -c, --classes: 类别数量（默认: 2）
    -v, --viz: 启用可视化显示
    -n, --no-save: 不保存输出文件
    --bilinear: 使用双线性插值上采样
    
注意:
    - 如果同时指定了 --output 和 --output-dir，--output 优先级更高
    - --output-dir 会自动创建不存在的文件夹
    - 使用 --output-dir 时，输出文件名会在原文件名基础上添加 _mask 后缀
"""
import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    """
    对单张图像进行预测
    
    参数:
        net: UNet神经网络模型
        full_img: PIL图像对象，待预测的完整图像
        device: 计算设备（CPU或CUDA）
        scale_factor: 图像缩放因子，用于调整输入图像大小
        out_threshold: 输出阈值，用于二分类时判断像素是否属于目标类别
    
    返回:
        numpy数组，预测的掩码图像
    """
    # 设置模型为评估模式，关闭dropout等训练特性
    net.eval()
    
    logging.debug(f'预处理图像，原始尺寸: {full_img.size}，缩放因子: {scale_factor}')
    
    # 预处理图像：归一化、缩放等操作
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)  # 添加batch维度 [C, H, W] -> [1, C, H, W]
    img = img.to(device=device, dtype=torch.float32)
    
    logging.debug(f'输入张量形状: {img.shape}')

    # 禁用梯度计算以节省内存和加速推理
    with torch.no_grad():
        output = net(img).cpu()  # 前向传播并将结果移到CPU
        logging.debug(f'模型输出形状: {output.shape}')
        
        # 将输出插值回原始图像大小
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        
        # 根据类别数选择不同的后处理方式
        if net.n_classes > 1:
            # 多分类：选择概率最大的类别
            mask = output.argmax(dim=1)
            logging.debug(f'多分类模式，类别数: {net.n_classes}')
        else:
            # 二分类：使用sigmoid激活函数和阈值
            mask = torch.sigmoid(output) > out_threshold
            logging.debug(f'二分类模式，阈值: {out_threshold}')

    return mask[0].long().squeeze().numpy()


def get_args():
    """
    解析命令行参数
    
    返回:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(description='使用训练好的模型对输入图像进行掩码预测')
    
    # 模型文件路径
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='指定存储模型的文件路径')
    
    # 输入图像文件（必需参数）
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', 
                        help='输入图像的文件名（可指定多个）', required=True)
    
    # 输出图像文件
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', 
                        help='输出图像的文件名（可指定多个）')
    
    # 输出文件夹
    parser.add_argument('--output-dir', metavar='DIR', type=str,
                        help='输出文件夹路径，所有掩码将保存到此文件夹（自动创建）')
    
    # 可视化选项
    parser.add_argument('--viz', '-v', action='store_true',
                        help='在处理时可视化图像')
    
    # 不保存输出
    parser.add_argument('--no-save', '-n', action='store_true', 
                        help='不保存输出的掩码图像')
    
    # 掩码阈值
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='将掩码像素视为白色的最小概率值（默认0.5）')
    
    # 图像缩放因子
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='输入图像的缩放因子（默认0.5）')
    
    # 双线性上采样
    parser.add_argument('--bilinear', action='store_true', default=False, 
                        help='使用双线性插值进行上采样')
    
    # 类别数量
    parser.add_argument('--classes', '-c', type=int, default=2, 
                        help='分类类别数量（默认2）')
    
    return parser.parse_args()


def get_output_filenames(args):
    """
    生成输出文件名
    优先级: --output > --output-dir > 自动生成（原文件名_OUT后缀）
    
    参数:
        args: 命令行参数对象
    
    返回:
        输出文件名列表
    """
    def _generate_name(fn):
        """为输入文件名生成对应的输出文件名"""
        return f'{os.path.splitext(fn)[0]}_OUT.png'
    
    def _generate_name_with_dir(fn, output_dir):
        """在指定文件夹中生成输出文件名"""
        # 获取原文件的基本名称（不含路径）
        basename = os.path.basename(fn)
        # 将后缀改为 _mask.png
        name_without_ext = os.path.splitext(basename)[0]
        return os.path.join(output_dir, f'{name_without_ext}_mask.png')

    # 优先级1: 如果指定了 --output 参数，直接使用
    if args.output:
        return args.output
    
    # 优先级2: 如果指定了 --output-dir 参数，在该文件夹中生成文件名
    if args.output_dir:
        # 创建输出文件夹（如果不存在）
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f'输出文件夹: {args.output_dir}')
        return list(map(lambda fn: _generate_name_with_dir(fn, args.output_dir), args.input))
    
    # 优先级3: 默认在原文件所在位置生成（添加_OUT后缀）
    return list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    """
    将预测的掩码数组转换为PIL图像对象
    
    参数:
        mask: numpy数组，预测的掩码
        mask_values: 掩码值列表，定义每个类别对应的像素值
    
    返回:
        PIL图像对象
    """
    # 根据mask_values的类型创建相应的输出数组
    if isinstance(mask_values[0], list):
        # RGB掩码：每个类别对应一个RGB值
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        # 二值掩码：使用布尔类型
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        # 灰度掩码：使用uint8类型
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    # 如果掩码是3维的（多通道），取最大值所在的通道
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    # 将类别索引映射到对应的像素值
    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    # 解析命令行参数
    args = get_args()
    
    # 配置日志输出格式
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    logging.info('=' * 50)
    logging.info('UNet图像分割预测程序启动')
    logging.info('=' * 50)

    # 获取输入和输出文件列表
    in_files = args.input
    out_files = get_output_filenames(args)
    
    logging.info(f'输入文件数量: {len(in_files)}')
    logging.info(f'输出文件数量: {len(out_files)}')
    
    # 如果指定了输出文件夹，显示文件夹路径
    if args.output_dir:
        logging.info(f'输出文件夹已创建: {os.path.abspath(args.output_dir)}')

    # 初始化UNet模型
    # n_channels: 输入图像通道数（RGB图像为3）
    # n_classes: 分类类别数
    # bilinear: 是否使用双线性插值进行上采样
    logging.info(f'初始化UNet模型 (通道数: 3, 类别数: {args.classes}, 双线性插值: {args.bilinear})')
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    # 选择计算设备（优先使用GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'正在加载模型: {args.model}')
    logging.info(f'使用设备: {device}')
    
    if device.type == 'cuda':
        logging.info(f'GPU设备名称: {torch.cuda.get_device_name(0)}')
        logging.info(f'GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')

    # 将模型移动到指定设备
    net.to(device=device)
    
    # 加载模型权重
    try:
        checkpoint = torch.load(args.model, map_location=device)
        
        # 判断加载的是完整checkpoint还是纯模型权重
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 完整checkpoint格式（包含optimizer、scheduler等）
            logging.info('检测到完整checkpoint格式')
            state_dict = checkpoint['model_state_dict']
            
            # 提取掩码值配置（如果存在），默认为[0, 1]
            mask_values = checkpoint.get('mask_values', [0, 1])
            
            # 显示checkpoint中的额外信息
            if 'epoch' in checkpoint:
                logging.info(f'训练轮次: {checkpoint["epoch"]}')
            if 'loss' in checkpoint:
                logging.info(f'训练损失: {checkpoint["loss"]:.4f}')
            if 'val_score' in checkpoint:
                logging.info(f'验证分数: {checkpoint["val_score"]:.4f}')
        else:
            # 纯模型权重格式
            logging.info('检测到纯模型权重格式')
            state_dict = checkpoint
            # 提取掩码值配置（如果存在），默认为[0, 1]
            mask_values = state_dict.pop('mask_values', [0, 1]) if 'mask_values' in state_dict else [0, 1]
        
        # 加载模型参数
        net.load_state_dict(state_dict)
        logging.info('✓ 模型加载完成！')
        logging.info(f'掩码值配置: {mask_values}')
    except Exception as e:
        logging.error(f'✗ 模型加载失败: {e}')
        raise
    
    logging.info(f'预测参数: 缩放因子={args.scale}, 阈值={args.mask_threshold}')
    logging.info('-' * 50)

    # 记录总体统计信息
    total_time = 0
    success_count = 0
    
    # 遍历所有输入图像进行预测
    for i, filename in enumerate(in_files):
        logging.info(f'[{i+1}/{len(in_files)}] 正在处理: {filename}')
        
        try:
            # 读取输入图像
            img = Image.open(filename)
            logging.info(f'  图像尺寸: {img.size[0]}x{img.size[1]}, 模式: {img.mode}')
            
            # 记录预测时间
            start_time = time.time()
            
            # 执行预测
            mask = predict_img(net=net,
                               full_img=img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)
            
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            logging.info(f'  预测耗时: {elapsed_time:.3f}秒')
            logging.info(f'  掩码形状: {mask.shape}, 唯一值: {np.unique(mask)}')

            # 保存预测结果
            if not args.no_save:
                out_filename = out_files[i]
                result = mask_to_image(mask, mask_values)
                result.save(out_filename)
                logging.info(f'  ✓ 掩码已保存到: {out_filename}')
            else:
                logging.info(f'  跳过保存（--no-save已启用）')

            # 可视化结果（如果启用）
            if args.viz:
                logging.info(f'  正在可视化结果，关闭窗口以继续...')
                plot_img_and_mask(img, mask)
            
            success_count += 1
            
        except Exception as e:
            logging.error(f'  ✗ 处理失败: {e}')
            continue
    
    # 输出总结信息
    logging.info('-' * 50)
    logging.info('预测完成！')
    logging.info(f'成功处理: {success_count}/{len(in_files)} 张图像')
    if success_count > 0:
        logging.info(f'总耗时: {total_time:.3f}秒')
        logging.info(f'平均耗时: {total_time/success_count:.3f}秒/张')
    logging.info('=' * 50)
