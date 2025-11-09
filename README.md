# Transformer (Encoder–Decoder) on IWSLT2017 En→De

基于 Transformer 模型实现的 IWSLT17 英德机器翻译系统。本项目提供了完整的训练、推理及评估流程，可用于英德双语翻译任务的研究与实践。

## 项目框架

```plaintext
transformer-iwslt17/
├── en-de.zip                   # 英德双语数据压缩包
├── example.txt                 # 翻译输入示例文本
├── example_trans.txt           # 翻译输出示例结果
├── requirements.txt            # 项目依赖库列表
├── scripts/ run.sh             # 一键运行脚本（包含环境配置、数据处理、训练、推理全流程）
├── en-de/                      # 英德双语数据
├── data/iwslt17/  				# IWSLT17官方原始数据          
├── results/                    # 实验结果与可视化文件
├── src/                        # 核心源代码目录
│   ├── data/                   # 数据处理相关代码
│   ├── models/                 # 模型定义相关代码
│   ├── train.py                # 模型训练脚本
│   ├── translate.py            # 翻译推理脚本
│   └── utils.py                # 工具函数脚本
├── checkpoints/                # 训练好的模型权重存放目录
└── Ablation_Experiment/        # 消融实验结果
```

## 权重说明

由于模型权重文件体积较大，未包含在本仓库中。如需获取预训练权重，请联系：`25125334@bjtu.edu.cn`

### 环境要求

- Python 3.8+
- 依赖库：见requirements.txt

### 运行流程

通过scripts/run.sh脚本可完成从环境配置到模型推理的全流程：

```bash
bash scripts/run.sh
```

**run.sh**

```bash
#!/usr/bin/env bash
set -e

# 1) 安装依赖
pip install -r requirements.txt

#2) 划分iwslt17/offical
python src/data/prepare_from_iwslt17_official.py \
  --root data/iwslt17/offical \
  --out_raw data/iwslt17/raw \
  --valid_year tst2013 \
  --test_year  tst2014

# 2) 准备数据与BPE
python src/data/prepare_data.py \
  --data_dir data/iwslt17 \
  --vocab_size 10000 \
  --src_lang en --tgt_lang de \
  --from_local

# 3) 训练
python -m src.train --data_dir data/iwslt17 --save_dir checkpoints --epochs 20 \
  --batch_size 32 --lr 1.0 --warmup_steps 4000 --d_model 512 --n_heads 8 --n_layers 6 \
  --d_ff 2048 --dropout 0.1 --label_smoothing 0.1 --max_len 100 --grad_clip 1.0 --weight_decay 0.01 \
  --seed 42 --bleu_max_samples 2000 --pos_enc sin --ln_style pre --optim adamw


# 4) 推理示例
echo "I am a postgraduate student from Beijing Jiaotong University." > example.txt
python -m src.translate \
  --data_dir data/iwslt17 \
  --model checkpoints/best.pt \
  --input example.txt --output example_trans.txt \
  --beam_size 8 --max_len 120
echo "Result:"
cat example_trans.txt

#5)BLEU 计算
echo "valid BLEU:"
python -m src.translate \
  --data_dir data/iwslt17 \
  --model checkpoints/best.pt \
  --input data/iwslt17/raw/valid.en \
  --output checkpoints/valid.hyp.de \
  --beam_size 8 --max_len 120

sacrebleu data/iwslt17/raw/valid.de -i checkpoints/valid.hyp.de -m bleu -b -w 2

echo "test BLEU:"
python -m src.translate \
  --data_dir data/iwslt17 \
  --model checkpoints/best.pt \
  --input data/iwslt17/raw/test.en \
  --output checkpoints/test.hyp.de \
  --beam_size 8 --max_len 120
  
sacrebleu data/iwslt17/raw/test.de -i checkpoints/test.hyp.de -m bleu -b -w 2
```

