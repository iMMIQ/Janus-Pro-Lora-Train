# 基于 Janus-Pro 的多模态微调与推理

本仓库提供基于 Janus-Pro 的多模态模型微调与推理脚本。本项目采用 LoRA 进行参数高效微调，支持图像-文本交互任务。

## 环境配置

运行脚本前请先配置环境：

```bash
pip install -r requirements.txt

git lfs install
git clone https://huggingface.co/deepseek-ai/Janus-Pro-1B

git clone https://github.com/deepseek-ai/Janus.git
mv Janus/janus janus
```

## 训练脚本：`train_janus_pro_lora.py`

### 功能说明
本脚本使用 LoRA 方法在图像-文本配对数据集上对 Janus-Pro 模型进行微调。

### 使用方式

```bash
python train_janus_pro_lora.py --data_dir dataset/images \
    --pretrained_model Janus-Pro-1B \
    --output_dir ./janus_lora_output \
    --batch_size 2 \
    --max_epochs 15 \
    --lr 5e-5 \
    --seed 42
```

### 参数说明
- `--data_dir`: 包含图片和文本文件的目录路径
- `--pretrained_model`: 预训练 Janus-Pro 模型名称或路径
- `--output_dir`: 微调模型保存目录
- `--batch_size`: 每批处理的样本数
- `--max_epochs`: 训练总轮数
- `--lr`: 学习率
- `--seed`: 随机种子（确保可复现性）

### 输出结果
微调后的模型（包含 LoRA 适配器和更新后的处理器）将保存在 `output_dir` 目录中。

---

## 推理脚本：`image2text.py`

### 功能说明
本脚本使用微调后的 Janus-Pro 模型（可选配 LoRA 适配器）进行图像到文本的推理。

### 使用方式

```bash
python image2text.py --model_path ./janus_lora_output \
    --image_path sample.jpg \
    --question "图片中有什么内容？" \
    --max_new_tokens 512
```

### 参数说明
- `--model_path`: 基础 Janus-Pro 模型路径
- `--lora_path`: （可选）微调后的 LoRA 适配器路径
- `--image_path`: 输入图片路径
- `--question`: 关于图片的文本询问
- `--max_new_tokens`: 生成文本的最大 token 数

### 输出结果
脚本将在控制台打印生成的文本响应。

---

更多细节请参考源代码文件 `train_janus_pro_lora.py` 和 `image2text.py`。