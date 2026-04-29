"""
ResNet-RS Additive Study
논문: Revisiting ResNets: Improved Training and Scaling Strategies (Bello et al., 2021)
데이터셋: STL-10 | 모델: ResNet-50 | 실험: baseline ~ exp10_resnetd 순차 누적 적용
"""

# ──────────────────────────────────────────────
# 0. Import
# ──────────────────────────────────────────────
# 파일/시스템 관련 유틸리티
import os               # 경로 생성, 파일 존재 여부 확인 등
import sys
import copy
import json
import time
import argparse

# 수치 연산 및 데이터 처리
import numpy as np
import pandas as pd

# 시각화 (서버 환경 대응)
import matplotlib
matplotlib.use('Agg')   # 서버 환경에서 디스플레이 없이 저장
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Pytorch 기본 모듈
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils
from torchsummary import summary

# ──────────────────────────────────────────────
# 1. Config
# ──────────────────────────────────────────────

# 모든 실험의 공통 기본값 (논문 baseline에 대응)
BASE_CONFIG = {
    "model":                 "resnet50",
    "dataset":               "STL10",
    "num_classes":           10,
    "image_size":            224,
    "batch_size":            32,
    "epochs":                90,        # baseline epoch
    "longer_epochs":         350,       # exp2_longer부터 적용할 epoch 수
    "optimizer":             "sgd",     # 논문: SGD Momentum 사용
    "lr":                    0.1,       # 초기 학습률
    "momentum":              0.9,       # SGD momentum 계수
    "weight_decay":          1e-4,      # L2 정규화 강도 (기본값)
    "use_cosine":            False,     # cosine LR decay 사용 여부
    "use_ema":               False,     # EMA(Exponential Moving Average) 사용 여부
    "ema_decay":             0.9999,    # EMA decay 계수
    "label_smoothing":       0.0,       # label smoothing 비율 (0이면 일반 CE)
    "stochastic_depth":      0.0,       # stochastic depth drop rate
    "use_randaugment":       False,     # RandAugment 사용 여부
    "randaugment_num_ops":   2,         # RandAugment 연산 수
    "randaugment_magnitude": 10,        # RandAugment 변환 강도
    "dropout":               0.0,       # FC 앞 dropout 비율
    "use_se":                False,     # Squeeze-and-Excitation 사용 여부
    "se_ratio":              0.25,      # SE block 채널 축소 비율
    "use_resnetd":           False,     # ResNet-D stem/downsample 사용 여부
    "seed":                  1,
}


def get_config(exp_name: str) -> dict:
    """
    실험 이름을 받아 해당 실험의 config를 반환.
    논문 Table 1의 additive study 순서로 누적 적용.
    """
    cfg = copy.deepcopy(BASE_CONFIG)    # 기본 config 깊은 복사 (실험 간 독립 보장)
    cfg["exp_name"] = exp_name
    cfg["save_dir"] = f"./results/{exp_name}"   # 실험 결과 저장 경로

    if exp_name == "baseline":
        # 논문: ResNet-200, 90 epochs, StepLR (우리: ResNet-50, 20 epochs, StepLR)
        pass

    elif exp_name == "exp1_cosine":
        # 논문: + Cosine LR Decay → +0.3%
        cfg["use_cosine"] = True        # StepLR → CosineAnnealingLR 교체

    elif exp_name == "exp2_longer":
        # 논문: + Increase training epochs (정규화 없이는 오히려 -0.5%)
        cfg["use_cosine"] = True
        cfg["epochs"] = cfg["longer_epochs"]    # 20 → 50 epoch으로 증가

    elif exp_name == "exp3_ema":
        # 논문: + EMA of weights → +0.3%
        cfg["use_cosine"] = True
        cfg["epochs"] = cfg["longer_epochs"]
        cfg["use_ema"] = True           # 학습 중 파라미터 EMA 유지

    elif exp_name == "exp4_label_smoothing":
        # 논문: + Label Smoothing → +1.3% (가장 큰 단독 기여)
        cfg["use_cosine"] = True
        cfg["epochs"] = cfg["longer_epochs"]
        cfg["use_ema"] = False
        cfg["label_smoothing"] = 0.1   # one-hot 대신 soft label 사용

    elif exp_name == "exp5_stochastic_depth":
        # 논문: + Stochastic Depth → +0.2%
        cfg["use_cosine"] = True
        cfg["epochs"] = cfg["longer_epochs"]
        cfg["use_ema"] = False
        cfg["label_smoothing"] = 0.1
        cfg["stochastic_depth"] = 0.1  # 각 residual block을 확률적으로 skip

    elif exp_name == "exp6_randaugment":
        # 논문: + RandAugment → +0.4%
        cfg["use_cosine"] = True
        cfg["epochs"] = cfg["longer_epochs"]
        cfg["use_ema"] = False
        cfg["label_smoothing"] = 0.1
        cfg["stochastic_depth"] = 0.1
        cfg["use_randaugment"] = True  # 랜덤 이미지 변환 augmentation 추가

    elif exp_name == "exp7_dropout":
        # 논문: + Dropout on FC → -0.3% (weight decay 감소 전에는 오히려 손해)
        cfg["use_cosine"] = True
        cfg["epochs"] = cfg["longer_epochs"]
        cfg["use_ema"] = False
        cfg["label_smoothing"] = 0.1
        cfg["stochastic_depth"] = 0.1
        cfg["use_randaugment"] = True
        cfg["dropout"] = 0.25          # global avg pool 직후 dropout 추가

    elif exp_name == "exp8_reduced_wd":
        # 논문: + Decrease weight decay → +1.5% (가장 큰 반전 포인트)
        cfg["use_cosine"] = True
        cfg["epochs"] = cfg["longer_epochs"]
        cfg["use_ema"] = False
        cfg["label_smoothing"] = 0.1
        cfg["stochastic_depth"] = 0.1
        cfg["use_randaugment"] = True
        cfg["dropout"] = 0.25
        cfg["weight_decay"] = 4e-5     # 1e-4 → 4e-5로 감소 (regularization 완화)

    elif exp_name == "exp9_se":
        # 논문: + Squeeze-and-Excitation → +0.7%
        cfg["use_cosine"] = True
        cfg["epochs"] = cfg["longer_epochs"]
        cfg["use_ema"] = False
        cfg["label_smoothing"] = 0.1
        cfg["stochastic_depth"] = 0.1
        cfg["use_randaugment"] = True
        cfg["dropout"] = 0.25
        cfg["weight_decay"] = 4e-5
        cfg["use_se"] = True           # 채널 간 상호작용으로 feature 재보정

    elif exp_name == "exp10_resnetd":
        # 논문: + ResNet-D → +0.5% (architecture 개선 마지막 단계)
        cfg["use_cosine"] = True
        cfg["epochs"] = cfg["longer_epochs"]
        cfg["use_ema"] = False
        cfg["label_smoothing"] = 0.1
        cfg["stochastic_depth"] = 0.1
        cfg["use_randaugment"] = True
        cfg["dropout"] = 0.25
        cfg["weight_decay"] = 4e-5
        cfg["use_se"] = True
        cfg["use_resnetd"] = True      # stem 3개 conv + downsample avg pool 구조

    else:
        raise ValueError(f"Unknown experiment name: {exp_name}")

    return cfg


# 실험 이름 목록 (논문 Table 1 순서와 동일)
EXP_LIST = [
    "baseline",
    "exp1_cosine",
    "exp2_longer",
    "exp3_ema",
    "exp4_label_smoothing",
    "exp5_stochastic_depth",
    "exp6_randaugment",
    "exp7_dropout",
    "exp8_reduced_wd",
    "exp9_se",
    "exp10_resnetd",
]


# ──────────────────────────────────────────────
# 2. Model
# ──────────────────────────────────────────────

# ── 추가 모듈 1: SEBlock ─────────────────────────────────────
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (Hu et al., 2018)
    채널별 중요도를 학습하여 feature map을 재보정.
    exp9_se부터 BottleneckBlock 내부에 삽입됨.
    """
    def __init__(self, channels, ratio=0.25):
        super().__init__()
        mid = max(1, int(channels * ratio))     # 축소된 중간 채널 수 계산
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                    # 공간 차원을 1x1로 압축 (Squeeze)
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),       # 채널 축소 FC
            nn.ReLU(),
            nn.Linear(mid, channels, bias=False),       # 채널 복원 FC
            nn.Sigmoid()                                # 0~1 사이 채널 중요도 출력 (Excitation)
        )

    def forward(self, x):
        scale = self.fc(x).view(x.size(0), x.size(1), 1, 1)    # (B, C) → (B, C, 1, 1)로 reshape
        return x * scale    # 채널별 중요도 가중치를 element-wise 곱으로 재보정


# ── 추가 모듈 2: StochasticDepth ─────────────────────────────
class StochasticDepth(nn.Module):
    """
    Stochastic Depth (Huang et al., 2016)
    학습 시 residual path 전체를 drop_prob 확률로 0으로 만들어 skip connection만 통과.
    inference 시에는 항상 전체 적용 (keep_prob로 기댓값 보정).
    exp5_stochastic_depth부터 BottleneckBlock 내부에 삽입됨.
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob  # block의 residual path를 skip할 확률

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x    # 평가 모드 or drop_prob=0이면 그대로 통과
        keep_prob = 1 - self.drop_prob
        # 배치 차원만 샘플링 (같은 배치 내 같은 block이 같이 drop됨)
        mask = torch.bernoulli(torch.full((x.size(0), 1, 1, 1), keep_prob, device=x.device))
        return x * mask / keep_prob     # drop된 경우 0, 아닌 경우 keep_prob으로 스케일 보정


# ── 기존 BasicBlock: 구조 그대로 유지 ───────────────────────
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride,
                          bias=False),      # 이미지 크기와 채널 개수를 맞춤 (합산 가능하도록)
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x


# ── 기존 BottleneckBlock: SE / StochasticDepth / ResNet-D만 추가 ──
class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1,
                 use_se=False, se_ratio=0.25,
                 drop_prob=0.0, use_resnetd=False):
        # 기존 인자(in_channels, out_channels, stride) 그대로 유지
        # 추가 인자: use_se, se_ratio, drop_prob, use_resnetd
        super().__init__()

        # 기존 residual_function 구조 그대로 유지
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleneckBlock.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleneckBlock.expansion)
        )

        # [추가] SE block: use_se=True이면 residual path 끝에 삽입, 아니면 Identity(통과)
        self.se = SEBlock(out_channels * BottleneckBlock.expansion, se_ratio) if use_se else nn.Identity()

        # [추가] Stochastic Depth: drop_prob>0이면 residual path 전체를 확률적으로 skip
        self.stoch_depth = StochasticDepth(drop_prob)

        # 기존 shortcut 구조 유지 + ResNet-D downsample 분기 추가
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleneckBlock.expansion:
            if use_resnetd and stride == 2:
                # [추가] ResNet-D: 기존 stride-2 1x1 conv → avgpool(2x2) + 1x1 conv로 교체
                # stride-2 conv의 정보 손실을 average pooling으로 완화
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True),  # 2x2 average pooling으로 다운샘플
                    nn.Conv2d(in_channels, out_channels * BottleneckBlock.expansion,
                              kernel_size=1, stride=1, bias=False),         # stride=1로 채널만 맞춤
                    nn.BatchNorm2d(out_channels * BottleneckBlock.expansion)
                )
            else:
                # 기존 projection shortcut 그대로
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * BottleneckBlock.expansion, kernel_size=1, stride=stride,
                              bias=False),      # 이미지 크기와 채널 개수를 맞춤 (합산 가능하도록)
                    nn.BatchNorm2d(out_channels * BottleneckBlock.expansion)
                )

    def forward(self, x):
        residual = self.residual_function(x)    # 기존: residual path 계산
        residual = self.se(residual)            # [추가] SE: 채널 중요도 재보정 (use_se=False면 통과)
        residual = self.stoch_depth(residual)   # [추가] Stochastic Depth: 확률적 skip (drop_prob=0이면 통과)
        x = residual + self.shortcut(x)         # 기존: skip connection 합산
        x = self.relu(x)                        # 기존: ReLU 활성화
        return x


# ── 기존 ResNet 클래스: stem/dropout 추가만, 나머지 구조 그대로 유지 ──
class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=10, init_weights=True,
                 dropout=0.0, use_se=False, se_ratio=0.25,
                 stochastic_depth=0.0, use_resnetd=False):
        # 기존 인자(block, num_block, num_classes, init_weights) 그대로 유지
        # 추가 인자: dropout, use_se, se_ratio, stochastic_depth, use_resnetd
        super().__init__()
        self.in_channels = 64

        # [추가] ResNet-D stem: use_resnetd=True이면 7x7 단일 conv → 3x3 conv 3개로 교체
        # 기존 stem은 7x7 conv 1개였으나, ResNet-D는 3x3 conv 3개로 세밀한 초기 feature 추출
        if use_resnetd:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),   # 112x112
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                    # 56x56
            )
        else:
            # 기존 stem 구조 그대로
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

        # 총 block 수: stochastic depth의 drop rate를 선형 증가시키기 위한 기준값
        total_blocks = sum(num_block)   # ResNet-50: 3+4+6+3=16

        # 기존 _make_layer 호출 구조 유지 + 추가 인자만 전달
        self.conv2_x = self._make_layer(block, 64,  num_block[0], stride=1,
                                        use_se=use_se, se_ratio=se_ratio,
                                        base_drop=stochastic_depth,
                                        total_blocks=total_blocks, block_offset=0,
                                        use_resnetd=use_resnetd)
        self.conv3_x = self._make_layer(block, 128, num_block[1], stride=2,
                                        use_se=use_se, se_ratio=se_ratio,
                                        base_drop=stochastic_depth,
                                        total_blocks=total_blocks, block_offset=num_block[0],
                                        use_resnetd=use_resnetd)
        self.conv4_x = self._make_layer(block, 256, num_block[2], stride=2,
                                        use_se=use_se, se_ratio=se_ratio,
                                        base_drop=stochastic_depth,
                                        total_blocks=total_blocks, block_offset=num_block[0]+num_block[1],
                                        use_resnetd=use_resnetd)
        self.conv5_x = self._make_layer(block, 512, num_block[3], stride=2,
                                        use_se=use_se, se_ratio=se_ratio,
                                        base_drop=stochastic_depth,
                                        total_blocks=total_blocks, block_offset=num_block[0]+num_block[1]+num_block[2],
                                        use_resnetd=use_resnetd)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # [추가] Dropout: global avg pool 직후, FC 직전에 삽입 (exp7부터, 기본값 0이면 비활성)
        self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Linear(512 * block.expansion, num_classes)     # fc layer

        # weights initialization
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride,
                    use_se=False, se_ratio=0.25,
                    base_drop=0.0, total_blocks=16, block_offset=0,
                    use_resnetd=False):
        # 기존: strides 리스트로 첫 block만 stride 적용하는 방식 그대로 유지
        strides = [stride] + [1] * (num_blocks - 1)     # 첫 번째 block은 다운샘플링, 나머지는 stride=1
        layers = []
        for i, s in enumerate(strides):
            # [추가] block 깊이에 비례한 drop rate 선형 증가 (얕은 층은 적게, 깊은 층은 많이 drop)
            drop_prob = base_drop * (block_offset + i) / (total_blocks - 1) if base_drop > 0 else 0.0
            # BasicBlock은 추가 인자 없이 기존 방식으로 생성, BottleneckBlock만 추가 인자 전달
            if block == BottleneckBlock:
                layers.append(block(self.in_channels, out_channels, s,
                                    use_se=use_se, se_ratio=se_ratio,
                                    drop_prob=drop_prob, use_resnetd=use_resnetd))
            else:
                layers.append(block(self.in_channels, out_channels, s))    # BasicBlock: 기존 그대로
            self.in_channels = out_channels * block.expansion   # 다음 block을 위해 입력 채널 수 업데이트
        return nn.Sequential(*layers)   # 리스트에 담긴 block들을 순차 연결

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)      # flatten
        x = self.dropout(x)             # [추가] dropout (exp7부터 활성화, 기본값 p=0이면 통과)
        output = self.fc(x)
        return output

    # ResNet 모델의 가중치 초기화 함수
    def _initialize_weights(self):
        for m in self.modules():        # 모델의 모든 모듈(레이어)을 순회
            if isinstance(m, nn.Conv2d):    # Conv2d 레이어인 경우
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # He 초기화 방법 적용
                if m.bias is not None:       # 편향이 존재하는 경우 편향 초기화
                    nn.init.constant_(m.bias, 0)   # 편향을 0으로 초기화
            elif isinstance(m, nn.BatchNorm2d):   # BatchNorm2d 레이어인 경우
                nn.init.constant_(m.weight, 1)     # 배치 정규화의 가중치를 1로 초기화
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)       # 배치 정규화의 편향을 0으로 초기화
            elif isinstance(m, nn.Linear):        # Linear 레이어인 경우
                nn.init.normal_(m.weight, 0, 0.01)   # 가중치를 평균 0, 표준편차 0.01인 정규분포로 초기화
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)           # 편향을 0으로 초기화

    # ResNet 정의 (기존 팩토리 메서드 그대로 유지)
    def resnet18():
        return ResNet(BasicBlock, [2, 2, 2, 2])

    def resnet34():
        return ResNet(BasicBlock, [3, 4, 6, 3])

    def resnet50(num_classes=10, dropout=0.0, use_se=False, se_ratio=0.25,
                 stochastic_depth=0.0, use_resnetd=False):
        # [수정] 추가 옵션 인자를 받아 ResNet에 전달 (기본값=False/0이면 기존 동작과 동일)
        return ResNet(BottleneckBlock, [3, 4, 6, 3],
                      num_classes=num_classes, dropout=dropout,
                      use_se=use_se, se_ratio=se_ratio,
                      stochastic_depth=stochastic_depth, use_resnetd=use_resnetd)

    def resnet101():
        return ResNet(BottleneckBlock, [3, 4, 23, 3])

    def resnet152():
        return ResNet(BottleneckBlock, [3, 8, 36, 3])


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    """config를 받아 resnet50 팩토리 메서드로 모델을 생성하고 device로 이동"""
    model = ResNet.resnet50(
        num_classes=cfg["num_classes"],
        dropout=cfg["dropout"],
        use_se=cfg["use_se"],
        se_ratio=cfg["se_ratio"],
        stochastic_depth=cfg["stochastic_depth"],
        use_resnetd=cfg["use_resnetd"],
    ).to(device)
    return model


# ──────────────────────────────────────────────
# 3. EMA (Exponential Moving Average)
# ──────────────────────────────────────────────

class ModelEMA:
    """
    모델 파라미터의 Exponential Moving Average 유지.
    학습 중 파라미터의 이동 평균을 별도로 보관하고, 평가 시 EMA 파라미터 사용.
    논문: ema_decay=0.9999 사용
    """
    def __init__(self, model: nn.Module, decay: float = 0.99):
        self.decay = decay
        # EMA 파라미터를 별도 저장 (gradient 계산 불필요)
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        """매 step 후 EMA 파라미터 업데이트: shadow = decay * shadow + (1-decay) * param"""
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.float()

    def apply_to(self, model: nn.Module):
        """평가 시 EMA 파라미터를 모델에 적용"""
        model.load_state_dict({k: v.to(next(model.parameters()).device)
                                for k, v in self.shadow.items()})


# ──────────────────────────────────────────────
# 4. Dataset & DataLoader
# ──────────────────────────────────────────────

DATA_PATH = '/home/undergraduate/20231372_TY/Pytorch/ResNet/rdata'  # 데이터셋 경로


def get_dataloaders(cfg: dict):
    """
    STL-10 데이터셋 로드, mean/std 계산, transform 적용, DataLoader 반환.
    train:val = 8:2 분할 (Subset 방식으로 transform 분리)
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset path not found: {DATA_PATH}")

    # ── mean/std 계산용 원본 로드 ────────────────────────────
    raw_ds = datasets.STL10(root=DATA_PATH, split='train',
                             download=False, transform=transforms.ToTensor())

    # 전체 train set으로 mean/std 계산 (채널별 평균)
    all_mean = np.mean([x.numpy().mean(axis=(1, 2)) for x, _ in raw_ds], axis=0)
    all_std  = np.mean([x.numpy().std(axis=(1, 2))  for x, _ in raw_ds], axis=0)
    mean = all_mean.tolist()    # [R_mean, G_mean, B_mean]
    std  = all_std.tolist()     # [R_std,  G_std,  B_std]

    print(f"  Mean: {[f'{v:.4f}' for v in mean]}  Std: {[f'{v:.4f}' for v in std]}")

    # ── Transform 정의 ──────────────────────────────────────
    aug_list = [transforms.Resize((cfg["image_size"], cfg["image_size"]))]

    if cfg["use_randaugment"]:
        # RandAugment: 랜덤 이미지 변환 n개를 magnitude 강도로 적용
        aug_list.append(transforms.RandAugment(
            num_ops=cfg["randaugment_num_ops"],
            magnitude=cfg["randaugment_magnitude"]
        ))
    aug_list += [
        transforms.RandomHorizontalFlip(),  # 기본 augmentation: 좌우 반전
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    train_transform = transforms.Compose(aug_list)

    # val/test는 augmentation 없이 정규화만 적용
    eval_transform = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ── Subset으로 train/val 분리 (transform 다르게 적용) ────
    train_base = datasets.STL10(root=DATA_PATH, split='train',
                                 download=False, transform=train_transform)
    val_base   = datasets.STL10(root=DATA_PATH, split='train',
                                 download=False, transform=eval_transform)

    num = len(train_base)
    idx = np.arange(num)
    rng = np.random.RandomState(cfg["seed"])    # 재현성 보장
    rng.shuffle(idx)

    train_size   = int(0.8 * num)               # 80% train
    train_idx    = idx[:train_size]             # 앞 80%: train
    val_idx      = idx[train_size:]             # 뒤 20%: val

    train_ds = Subset(train_base, train_idx)    # train transform 적용된 subset
    val_ds   = Subset(val_base,   val_idx)      # eval transform 적용된 subset
    test_ds  = datasets.STL10(root=DATA_PATH, split='test',
                                download=False, transform=eval_transform)

    print(f"  train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                          num_workers=0, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,
                          num_workers=0, pin_memory=True)

    return train_dl, val_dl, test_dl


# ──────────────────────────────────────────────
# 5. Train / Eval utilities
# ──────────────────────────────────────────────

def run_epoch(model, loader, loss_fn, device, opt=None, ema=None):
    """
    한 epoch 동안 학습 또는 평가를 수행하는 공통 함수.
    opt가 주어지면 학습 모드(역전파 포함), 없으면 평가 모드.
    """
    is_train = (opt is not None)
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_n = 0.0, 0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)                     # forward pass
            loss = loss_fn(out, yb)             # loss 계산

            if is_train:
                opt.zero_grad()                 # 이전 gradient 초기화
                loss.backward()                 # 역전파
                opt.step()                      # 파라미터 업데이트
                if ema is not None:
                    ema.update(model)           # EMA 파라미터 업데이트

            total_loss    += loss.item() * xb.size(0)       # 배치 loss 누적
            total_correct += out.argmax(1).eq(yb).sum().item()  # 정답 수 누적
            total_n       += xb.size(0)

    avg_loss = total_loss / total_n         # 평균 loss
    avg_acc  = total_correct / total_n     # 평균 정확도
    return avg_loss, avg_acc


def evaluate_test(model, test_dl, loss_fn, device, class_names):
    """
    테스트 세트 평가: 전체 loss, 전체 accuracy, 클래스별 accuracy 계산.
    best_model.pt 로드 후 1회만 호출 (test leakage 방지).
    """
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            out  = model(xb)
            loss = loss_fn(out, yb)

            total_loss    += loss.item() * xb.size(0)
            total_correct += out.argmax(1).eq(yb).sum().item()
            total_n       += xb.size(0)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_targets.extend(yb.cpu().numpy())

    avg_loss = total_loss / total_n
    avg_acc  = total_correct / total_n

    # 클래스별 정확도 계산
    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)
    class_accs  = {}
    for i, name in enumerate(class_names):
        mask = (all_targets == i)
        class_accs[name] = float((all_preds[mask] == i).sum()) / mask.sum()

    return avg_loss, avg_acc, class_accs


# ──────────────────────────────────────────────
# 6. Save utilities
# ──────────────────────────────────────────────

CLASS_NAMES = ['airplane', 'bird', 'car', 'cat', 'deer',
               'dog', 'horse', 'monkey', 'ship', 'truck']


def save_result_log(save_dir, cfg, best_val_loss, best_val_acc, best_epoch,
                    test_loss, test_acc, elapsed_min):
    """result_log.txt: 논문 Table 1 행 1개에 대응하는 실험 결과 기록"""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'result_log.txt'), 'w') as f:
        f.write(f"Experiment      : {cfg['exp_name']}\n")
        f.write(f"Epochs          : {cfg['epochs']}\n")
        f.write(f"Optimizer       : {cfg['optimizer']}  lr={cfg['lr']}  wd={cfg['weight_decay']}\n")
        f.write(f"Cosine LR       : {cfg['use_cosine']}\n")
        f.write(f"EMA             : {cfg['use_ema']}\n")
        f.write(f"Label Smoothing : {cfg['label_smoothing']}\n")
        f.write(f"Stoch. Depth    : {cfg['stochastic_depth']}\n")
        f.write(f"RandAugment     : {cfg['use_randaugment']}\n")
        f.write(f"Dropout         : {cfg['dropout']}\n")
        f.write(f"SE              : {cfg['use_se']}\n")
        f.write(f"ResNet-D        : {cfg['use_resnetd']}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best val loss   : {best_val_loss:.6f}\n")
        f.write(f"Best val acc    : {best_val_acc * 100:.2f}%\n")
        f.write(f"Best epoch      : {best_epoch}\n")
        f.write(f"Test loss       : {test_loss:.6f}\n")
        f.write(f"Test accuracy   : {test_acc * 100:.2f}%\n")
        f.write(f"Training time   : {elapsed_min:.2f} min\n")


def save_history_csv(save_dir, history):
    """history.csv: epoch별 loss/acc/lr 기록 (논문 재현 분석용)"""
    df = pd.DataFrame(history, columns=['epoch', 'train_loss', 'val_loss',
                                         'train_acc', 'val_acc', 'lr'])
    df.to_csv(os.path.join(save_dir, 'history.csv'), index=False)


def save_config_json(save_dir, cfg):
    """config.json: 실험 설정 전체를 JSON으로 저장 (재현성 보장)"""
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)


def plot_loss_acc(save_dir, history, exp_name):
    """Train vs Val loss / accuracy 그래프 저장"""
    df = pd.DataFrame(history, columns=['epoch', 'train_loss', 'val_loss',
                                         'train_acc', 'val_acc', 'lr'])
    epochs = df['epoch'].tolist()

    # Loss 그래프
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df['train_loss'], label='Train Loss')
    plt.plot(epochs, df['val_loss'],   label='Val Loss')
    plt.title(f"Train-Val Loss  [{exp_name}]")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_progress.png'), dpi=150)
    plt.close()

    # Accuracy 그래프
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [v * 100 for v in df['train_acc']], label='Train Acc')
    plt.plot(epochs, [v * 100 for v in df['val_acc']],   label='Val Acc')
    plt.title(f"Train-Val Accuracy  [{exp_name}]")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_progress.png'), dpi=150)
    plt.close()


def plot_class_accuracy(save_dir, class_accs, overall_acc, exp_name):
    """클래스별 test accuracy 막대 그래프 저장"""
    names = list(class_accs.keys())
    vals  = [class_accs[n] * 100 for n in names]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, vals, color='steelblue', edgecolor='white')
    # 각 막대 위에 수치 표시
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    plt.title(f"Class-wise Test Accuracy  [{exp_name}]  Overall: {overall_acc*100:.2f}%")
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 110)
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_class_accuracy.png'), dpi=150)
    plt.close()


# ──────────────────────────────────────────────
# 7. Summary plot (전체 실험 비교)
# ──────────────────────────────────────────────

def plot_additive_summary(results: list):
    """
    논문 Table 1에 대응하는 실험별 성능 비교 그래프.
    results: [{'exp': str, 'test_acc': float}, ...]
    """
    os.makedirs('./results', exist_ok=True)

    exp_labels = [r['exp'] for r in results]
    accs       = [r['test_acc'] * 100 for r in results]
    deltas     = [0.0] + [accs[i] - accs[i-1] for i in range(1, len(accs))]

    # ── 1. 누적 성능 라인 차트 ──────────────────────────────
    plt.figure(figsize=(13, 5))
    plt.plot(range(len(accs)), accs, marker='o', linewidth=2, color='steelblue')
    for i, (a, d) in enumerate(zip(accs, deltas)):
        color = 'green' if d >= 0 else 'red'
        sign  = '+' if d >= 0 else ''
        plt.annotate(f'{a:.1f}%\n({sign}{d:.1f})',
                     xy=(i, a), xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=7.5, color=color)
    plt.xticks(range(len(exp_labels)), exp_labels, rotation=20, ha='right', fontsize=8)
    plt.ylabel('Test Accuracy (%)')
    plt.title('Additive Study: Test Accuracy per Experiment  (논문 Table 1 대응)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('./results/additive_accuracy_curve.png', dpi=150)
    plt.close()

    # ── 2. Δ 막대 차트 (각 실험의 개별 기여도) ──────────────
    colors = ['green' if d >= 0 else 'red' for d in deltas]
    plt.figure(figsize=(13, 4))
    bars = plt.bar(exp_labels, deltas, color=colors, edgecolor='white')
    for bar, d in zip(bars, deltas):
        sign = '+' if d >= 0 else ''
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.05 if d >= 0 else -0.15),
                 f'{sign}{d:.2f}%', ha='center', va='bottom', fontsize=8)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=20, ha='right', fontsize=8)
    plt.ylabel('Δ Test Accuracy (%)')
    plt.title('Per-Experiment Accuracy Change  (이전 단계 대비)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('./results/additive_delta_chart.png', dpi=150)
    plt.close()

    # ── 3. 논문 Table 1 형식 텍스트 파일 저장 ────────────────
    with open('./results/additive_table.txt', 'w') as f:
        f.write(f"{'Experiment':<28} {'Test Acc (%)':>12} {'Δ':>8}\n")
        f.write("-" * 52 + "\n")
        for r, d in zip(results, deltas):
            sign = '+' if d >= 0 else ''
            f.write(f"{r['exp']:<28} {r['test_acc']*100:>11.2f}% {sign+f'{d:.2f}%':>8}\n")
    print("\n[Summary] Saved: additive_accuracy_curve.png / additive_delta_chart.png / additive_table.txt")


# ──────────────────────────────────────────────
# 8. Main train loop
# ──────────────────────────────────────────────

def train_experiment(cfg: dict, device: torch.device) -> dict:
    """
    단일 실험(cfg)을 처음부터 끝까지 실행.
    학습 → val 기준 best model 저장 → test 평가 → 결과 저장
    """
    exp_name = cfg["exp_name"]
    save_dir = cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Experiment: {exp_name}")
    print(f"{'='*60}")

    # 재현성 시드 고정
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])

    # ── 데이터 로드 ─────────────────────────────────────────
    train_dl, val_dl, test_dl = get_dataloaders(cfg)

    # ── 모델 생성 ───────────────────────────────────────────
    model = build_model(cfg, device)
    save_config_json(save_dir, cfg)     # 실험 설정 저장

    # ── Loss function ───────────────────────────────────────
    # label_smoothing=0이면 일반 CE, >0이면 soft label 적용
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    # ── Optimizer ───────────────────────────────────────────
    # 논문: SGD Momentum (Adam 대신 사용, 논문 권장)
    opt = optim.SGD(model.parameters(),
                    lr=cfg["lr"],
                    momentum=cfg["momentum"],
                    weight_decay=cfg["weight_decay"])

    # ── LR Scheduler ────────────────────────────────────────
    if cfg["use_cosine"]:
        # 논문 권장: cosine annealing (T_max=전체 epoch)
        scheduler = CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=0)
    else:
        # baseline: StepLR (30 epoch마다 0.1배 감소)
        scheduler = StepLR(opt, step_size=30, gamma=0.1)

    # ── EMA 설정 ────────────────────────────────────────────
    ema = ModelEMA(model, decay=cfg["ema_decay"]) if cfg["use_ema"] else None

    path_best = os.path.join(save_dir, 'best_model.pt')    # best model 저장 경로
    best_val_loss = float('inf')
    best_val_acc  = 0.0
    best_epoch    = 0
    history       = []      # epoch별 기록 리스트
    start_time    = time.time()

    # ── Training loop ───────────────────────────────────────
    for epoch in range(1, cfg["epochs"] + 1):

        epoch_start = time.time()   # 시간 측정 시작 1 (epoch당 걸리는 시간)
        current_lr = opt.param_groups[0]['lr']  # 현재 학습률 확인

        # 학습 1 epoch
        train_loss, train_acc = run_epoch(model, train_dl, loss_fn, device, opt=opt, ema=ema)

        # 평가: EMA 사용 시 EMA 파라미터로 평가
        if ema is not None:
            ema_model = copy.deepcopy(model)    # 원본 모델 복사
            ema.apply_to(ema_model)             # EMA 파라미터 적용
            val_loss, val_acc = run_epoch(ema_model, val_dl, loss_fn, device)
        else:
            val_loss, val_acc = run_epoch(model, val_dl, loss_fn, device)

        scheduler.step()    # LR 스케줄러 업데이트

        history.append([epoch, train_loss, val_loss, train_acc, val_acc, current_lr])

        # val_loss 기준으로 best model 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            best_epoch    = epoch
            # EMA 사용 시 EMA 파라미터로 저장
            save_target = ema_model if ema is not None else model
            torch.save(save_target.state_dict(), path_best)

        elapsed = (time.time() - start_time) / 60
        epoch_time = (time.time() - epoch_start) / 60   # ✅ 추가
        print(f"  Epoch {epoch:>3}/{cfg['epochs']}  "
              f"lr={current_lr:.5f}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc*100:.2f}% "
              f"epoch_time={epoch_time:.2f}min  total={elapsed:.1f}min")

    # ── Test evaluation ─────────────────────────────────────
    # best model 로드 (test는 학습 종료 후 1회만 수행)
    best_model = build_model(cfg, device)
    best_model.load_state_dict(torch.load(path_best, map_location=device))
    test_loss, test_acc, class_accs = evaluate_test(
        best_model, test_dl, loss_fn, device, CLASS_NAMES)

    elapsed_min = (time.time() - start_time) / 60

    print(f"\n  [TEST] loss={test_loss:.4f}  acc={test_acc*100:.2f}%  "
          f"total_time={elapsed_min:.1f}min")

    # ── 결과 저장 ────────────────────────────────────────────
    save_history_csv(save_dir, history)
    save_result_log(save_dir, cfg, best_val_loss, best_val_acc, best_epoch,
                    test_loss, test_acc, elapsed_min)
    plot_loss_acc(save_dir, history, exp_name)
    plot_class_accuracy(save_dir, class_accs, test_acc, exp_name)

    print(f"  Saved to: {save_dir}/")

    return {"exp": exp_name, "test_acc": test_acc, "test_loss": test_loss,
            "best_val_acc": best_val_acc, "best_epoch": best_epoch}


# ──────────────────────────────────────────────
# 9. Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ResNet-RS Additive Study")
    parser.add_argument(
        '--exp', type=str, default='all',
        help=(f"실험 이름. 'all'이면 전체 실행. "
              f"선택지: all, {', '.join(EXP_LIST)}")
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # 실행할 실험 목록 결정
    if args.exp == 'all':
        run_list = EXP_LIST     # 전체 실험 순차 실행
    else:
        if args.exp not in EXP_LIST:
            print(f"[ERROR] Unknown experiment: {args.exp}")
            print(f"  선택 가능: {EXP_LIST}")
            sys.exit(1)
        run_list = [args.exp]   # 단일 실험만 실행

    all_results = []    # 전체 실험 결과 누적 (summary용)

    # 이미 완료된 실험 결과 불러오기 (all 실행 시 이어서 summary 생성)
    if args.exp == 'all':
        for exp in EXP_LIST:
            log_path = f"./results/{exp}/result_log.txt"
            if os.path.exists(log_path) and exp not in run_list:
                # 이미 결과가 있으면 파일에서 test_acc 파싱
                with open(log_path) as f:
                    for line in f:
                        if 'Test accuracy' in line:
                            acc = float(line.split(':')[1].strip().replace('%', '')) / 100
                            all_results.append({"exp": exp, "test_acc": acc})

    # 실험 실행
    for exp_name in run_list:
        cfg    = get_config(exp_name)
        result = train_experiment(cfg, device)
        all_results.append(result)

    # 전체 실험 완료 후 summary 그래프 생성
    # all_results가 EXP_LIST 순서와 맞는 경우에만 생성
    if set(r['exp'] for r in all_results) == set(EXP_LIST):
        # EXP_LIST 순서로 정렬
        ordered = sorted(all_results, key=lambda r: EXP_LIST.index(r['exp']))
        plot_additive_summary(ordered)

    print("\n" + "="*60)
    print("  모든 실험 완료")
    print("="*60)


if __name__ == '__main__':
    main()