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