# UNet训练参数推荐指南

本指南提供了针对不同硬件配置、数据集规模和任务类型的详细参数推荐。

## 📊 硬件配置检测

首先检测您的硬件配置：

```bash
# 检测GPU信息
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}'); print(f'当前GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无GPU\"}'); print(f'显存大小: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB' if torch.cuda.is_available() else '无GPU')"
```

## 🖥️ 按硬件配置推荐

### 1. 高端GPU配置 (RTX 4090/3090, A100等)
**显存**: 24GB+ | **推荐用途**: 大规模数据集训练

```bash
# 高性能训练配置
python train.py \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --amp \
    --scale 0.75 \
    --validation 10 \
    --classes 2

# 超大规模训练配置
python train.py \
    --epochs 200 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --amp \
    --scale 1.0 \
    --validation 5 \
    --classes 5
```

### 2. 中端GPU配置 (RTX 4080/3080, RTX 4070/3070等)
**显存**: 12-16GB | **推荐用途**: 标准数据集训练

```bash
# 标准训练配置
python train.py \
    --epochs 80 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --amp \
    --scale 0.5 \
    --validation 15 \
    --classes 2

# 多分类任务配置
python train.py \
    --epochs 100 \
    --batch-size 6 \
    --learning-rate 1e-4 \
    --amp \
    --scale 0.5 \
    --validation 15 \
    --classes 5
```

### 3. 入门级GPU配置 (RTX 4060/3060, GTX 1660等)
**显存**: 6-8GB | **推荐用途**: 小规模数据集训练

```bash
# 基础训练配置
python train.py \
    --epochs 50 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --amp \
    --scale 0.25 \
    --validation 20 \
    --classes 2

# 显存优化配置
python train.py \
    --epochs 60 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --amp \
    --bilinear \
    --scale 0.25 \
    --validation 20 \
    --classes 2
```

### 4. 低显存GPU配置 (RTX 3050, GTX 1050等)
**显存**: 4-6GB | **推荐用途**: 轻量级训练

```bash
# 轻量级训练配置
python train.py \
    --epochs 40 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --amp \
    --bilinear \
    --scale 0.125 \
    --validation 25 \
    --classes 2

# 最小显存配置
python train.py \
    --epochs 30 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --amp \
    --bilinear \
    --scale 0.1 \
    --validation 30 \
    --classes 2
```

### 5. CPU训练配置
**推荐用途**: 测试和调试

```bash
# CPU训练配置
python train.py \
    --epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --scale 0.25 \
    --validation 30 \
    --classes 2
```

## 📚 按数据集规模推荐

### 小数据集 (< 1000张图片)
```bash
python train.py \
    --epochs 100 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --amp \
    --scale 0.5 \
    --validation 30 \
    --classes 2
```

### 中等数据集 (1000-10000张图片)
```bash
python train.py \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --amp \
    --scale 0.5 \
    --validation 15 \
    --classes 2
```

### 大数据集 (> 10000张图片)
```bash
python train.py \
    --epochs 30 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --amp \
    --scale 0.75 \
    --validation 10 \
    --classes 2
```

## 🎯 按任务类型推荐

### 医学图像分割
```bash
python train.py \
    --epochs 80 \
    --batch-size 4 \
    --learning-rate 1e-5 \
    --amp \
    --scale 0.5 \
    --validation 20 \
    --classes 2
```

### 卫星图像分割
```bash
python train.py \
    --epochs 60 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --amp \
    --scale 0.75 \
    --validation 15 \
    --classes 5
```

### 细胞分割
```bash
python train.py \
    --epochs 100 \
    --batch-size 6 \
    --learning-rate 1e-4 \
    --amp \
    --scale 0.5 \
    --validation 15 \
    --classes 3
```

### 自动驾驶场景分割
```bash
python train.py \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --amp \
    --scale 0.75 \
    --validation 10 \
    --classes 10
```

## 🔧 参数调优策略

### 学习率调优
```bash
# 高学习率（快速收敛，可能不稳定）
--learning-rate 1e-3

# 标准学习率（推荐）
--learning-rate 1e-4

# 低学习率（稳定但慢）
--learning-rate 1e-5
```

### 批次大小调优
```bash
# 大批次（稳定但需要更多显存）
--batch-size 16

# 中等批次（平衡）
--batch-size 8

# 小批次（显存友好）
--batch-size 4
```

### 图像尺寸调优
```bash
# 高分辨率（精度高，显存需求大）
--scale 1.0

# 标准分辨率（推荐）
--scale 0.5

# 低分辨率（显存友好）
--scale 0.25
```

## 🚀 性能优化配置

### 最大化训练速度
```bash
python train.py \
    --epochs 50 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --amp \
    --scale 0.75 \
    --validation 10 \
    --classes 2
```

### 最大化显存效率
```bash
python train.py \
    --epochs 60 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --amp \
    --bilinear \
    --scale 0.25 \
    --validation 20 \
    --classes 2
```

### 最大化分割精度
```bash
python train.py \
    --epochs 100 \
    --batch-size 8 \
    --learning-rate 1e-5 \
    --amp \
    --scale 1.0 \
    --validation 15 \
    --classes 2
```

## 📈 训练监控和调试

### 快速测试配置
```bash
python train.py \
    --epochs 2 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --amp \
    --scale 0.25 \
    --validation 30 \
    --classes 2
```

### 调试配置
```bash
python train.py \
    --epochs 5 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --scale 0.25 \
    --validation 20 \
    --classes 2
```

## 🔄 从检查点恢复训练

```bash
# 从最新检查点恢复
python train.py \
    --load checkpoints/checkpoint_epoch10.pth \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --amp
```

## ⚠️ 常见问题解决

### 显存溢出
```bash
# 解决方案1：启用混合精度和减小批次
python train.py --amp --batch-size 1 --scale 0.25

# 解决方案2：使用双线性上采样
python train.py --bilinear --batch-size 2 --scale 0.25

# 解决方案3：组合优化
python train.py --amp --bilinear --batch-size 1 --scale 0.125
```

### 训练速度慢
```bash
# 启用混合精度训练
python train.py --amp --batch-size 8

# 增大批次大小
python train.py --batch-size 16 --amp

# 减小图像尺寸
python train.py --scale 0.75 --batch-size 8
```

### 分割精度低
```bash
# 增加训练轮数
python train.py --epochs 100

# 使用更高分辨率
python train.py --scale 1.0

# 调整学习率
python train.py --learning-rate 1e-5
```

## 📝 参数选择建议

1. **首先确定硬件限制**：根据显存大小选择batch_size和scale
2. **选择合适的学习率**：1e-4是很好的起点
3. **启用混合精度**：除非遇到数值稳定性问题
4. **调整验证集比例**：小数据集用更大比例
5. **监控训练过程**：使用WandB查看训练曲线
6. **定期保存检查点**：防止训练中断丢失进度

## 🎛️ 高级参数组合

### 多GPU训练准备（需要修改代码）
```bash
python train.py \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --amp \
    --scale 1.0 \
    --validation 10 \
    --classes 2
```

### 实验性配置
```bash
python train.py \
    --epochs 200 \
    --batch-size 8 \
    --learning-rate 5e-5 \
    --amp \
    --scale 0.75 \
    --validation 15 \
    --classes 2
```

记住：最佳参数需要根据具体的数据集和任务进行调整。建议从小配置开始，逐步优化。
