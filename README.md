# Pytorch 모델 구현

## Transformer

### Transformer.py

- Transformer 깡통으로 번역모델 구현
- 데이터셋: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=126

## GPT2

### GPT2_ko.py

- 한국어 데이터 기반 모델 학습
- 데이터셋: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=126

### GPT2_en.py + Finetuning

- 영어 next token prediction 모델 학습 (Alphaca finetuning을 위해 영어 데이터셋 사용)
- 데이터셋: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=126

#### Finetuning

- 데이터셋: https://huggingface.co/datasets/yahma/alpaca-cleaned
- 학습 방법: LoRA + DPO (억LoRA 맞습니다.)

## BERT

- 한국어 데이터 기반 모델 학습
- 데이터셋: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=126
  (이어지는 문장의 데이터셋이 아니라서 Masked Token Prediction 만 학습)

## ViT

- Vision Transformer
- 데이터셋: CIFAR10

## Swin Transformer

- Swin Transformer
- 데이터셋: CIFAR10

## ConvNeXt

- ConvNeXt
- 데이터셋: CIFAR10

## CLIP

## Mamba

## BMT

## AudioCLIP
