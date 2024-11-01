# Pytorch 모델 구현

## Transformer

- Transformer 깡통으로 번역모델 구현
- 데이터셋: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=126

## GPT2

### GPT2_ko.py

- 한국어 데이터 기반 모델 학습
- 데이터셋: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=126

### GPT2_en.py + Finetuning

- 영어 데이터 기반 모델 학습
- 데이터셋: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=126

#### Finetuning

- 데이터셋: https://huggingface.co/datasets/yahma/alpaca-cleaned
- 모델: GPT2_en.py
- 학습 방법: LoRA + DPO

## BERT

### BERT_ko.py

- 한국어 데이터 기반 모델 학습
- 데이터셋: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=126
  (이어지는 문장의 데이터셋이 아니라서 Next Sentence Prediction 유기)

## ViT

## Swin Transformer

## CLIP

## Mamba

## BMT

## AudioCLIP
