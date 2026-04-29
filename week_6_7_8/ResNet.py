# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim
from torch.optim.lr_scheduler import StepLR

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
# 추가: 학습/검증 데이터셋 분할
from torch.utils.data import DataLoader, Subset
import os

# display images
from torchvision import utils
import matplotlib.pyplot as plt

# utils
import numpy as np
import time
import copy

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# GPU 이름 출력
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Load Dataset : STL-10 - 태영이 다운 데이터셋 사용
# specify the data path
DATA_PATH = '/home/undergraduate/20231372_TY/Pytorch/ResNet/rdata'
path2data = DATA_PATH

if not os.path.exists(path2data):
    raise FileNotFoundError(f"Dataset path not found: {path2data}")

# load dataset
full_train_ds = datasets.STL10(root=path2data, split='train', download=False, transform=transforms.ToTensor())

# Dataset: calculate mean & std
# To normalize the dataset, calculate the mean and std
train_meanRGB = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in full_train_ds]
train_stdRGB = [np.std(x.numpy(), axis=(1, 2)) for x, _ in full_train_ds]

train_meanR = np.mean([m[0] for m in train_meanRGB])
train_meanG = np.mean([m[1] for m in train_meanRGB])
train_meanB = np.mean([m[2] for m in train_meanRGB])

train_strR = np.mean([s[0] for s in train_stdRGB])
train_strG = np.mean([s[1] for s in train_stdRGB])
train_strB = np.mean([s[2] for s in train_stdRGB])

print(f"train - Mean RGB: ({train_meanR:.4f}, {train_meanG:.4f}, {train_meanB:.4f})")
print(f"train - Std RGB: ({train_strR:.4f}, {train_strG:.4f}, {train_strB:.4f})")

# display some images
def show(img, y=None, color=True, save_path=None):              # y: label
    npimg = img.numpy()                         # tensor -> numpy
    npimg_tr = np.transpose(npimg, (1, 2, 0))   # 차원 순서 변경: (C, H, W) -> (H, W, C)
    plt.figure(figsize=(12, 3))
    plt.imshow(npimg_tr)                        # numpy 배열을 화면에 이미지 표시

    if y is not None:                           # 레이블이 제공된 경우, 이미지 위에 레이블로 표시
        plt.title("Label: " + str(y))
    plt.axis('off')

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved image to {save_path}")

    plt.close()

# 랜덤 고정
np.random.seed(1)
torch.manual_seed(1)

# train dataset에서 8개 데이터 랜덤하게 선정
grid_size = 8                                                   # np.random.randint(low, high, size) : low 이상 high 미만의 정수 중에서 size 개수만큼 랜덤하게 선택
rnd_inds = np.random.randint(0, len(full_train_ds), size=grid_size)
print('image indices: ', rnd_inds)

# 샘플 이미지와 라벨 추출
x_grid = [train_ds[i][0] for i in rnd_inds]    # 데이터셋: 이미지 데이터
y_grid = [train_ds[i][1] for i in rnd_inds]    # 데이터셋: 라벨

# 이미지들을 그리드 형태로 결합 (padding: 이미지 사이 간격(픽셀))
x_grid = utils.make_grid(x_grid, nrow=grid_size, padding=2)    # nrow: 한 행에 배치할 이미지 수

save_dir = './visualization/stl10_samples'
save_path = os.path.join(save_dir, 'sample_grid.png')

show(x_grid, y_grid, save_path=save_path)


# define the image transformation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[train_meanR, train_meanG, train_meanB], 
                         std=[train_strR, train_strG, train_strB])
    
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[train_meanR, train_meanG, train_meanB], 
                         std=[train_strR, train_strG, train_strB])
])

# 같은 STL10 train split을 transform만 다르게 두 번 로드
train_base_ds = datasets.STL10(
    root=path2data,
    split='train',
    download=False,
    transform=train_transform
)

val_base_ds = datasets.STL10(
    root=path2data,
    split='train',
    download=False,
    transform=eval_transform
)

test_ds = datasets.STL10(
    root=path2data,
    split='test',
    download=False,
    transform=eval_transform
)

# 5000개 train split을 8:2로 나눌 index 생성
num_train = len(train_base_ds)
indices = np.arange(num_train)

# 랜덤 고정
np.random.seed(1)
# 랜덤하게 순서를 섞음
np.random.shuffle(indices)

# train할 사이즈(인덱싱), val할 사이즈 인덱싱
train_size = int(0.8 * num_train)

train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_ds = Subset(train_base_ds, train_indices)
val_ds = Subset(val_base_ds, val_indices)

print(f"Number of training samples: {len(train_ds)}")
print(f"Number of validation samples: {len(val_ds)}")
print(f"Number of test samples: {len(test_ds)}")

# create dataloader
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)
# 테스트 DataLoader 생성 (shuffle=False: 평가 시 순서 유지)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)


print("-" * 50)

# Basic Block & Bottleneck Block
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

class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

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
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleneckBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleneckBlock.expansion, kernel_size=1, stride=stride, 
                          bias=False),      # 이미지 크기와 채널 개수를 맞춤 (합산 가능하도록)
                nn.BatchNorm2d(out_channels * BottleneckBlock.expansion)
            )
    
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x
    

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=10, init_weights=True):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # block, out_channels, num_block, stride=다운샘플링
        self.conv2_x = self._make_layer(block, 64, num_block[0], stride=1)      # 1st layer
        self.conv3_x = self._make_layer(block, 128, num_block[1], stride=2)     # 2nd layer
        self.conv4_x = self._make_layer(block, 256, num_block[2], stride=2)     # 3rd layer
        self.conv5_x = self._make_layer(block, 512, num_block[3], stride=2)     # 4th layer

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)                 # fc layer

        # weights initialization
        if init_weights:
            self._initialize_weights()
    
    # 하나의 ResNet layer를 생성하는 함수 (여러 개의 residual block으로 쌓음)
    # param: resibual block 종류, layer 채널 수, block 개수, stride=다운샘플링
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)     # 첫 번째 block은 다운샘플링, 나머지는 stride=1 예시: [2]+[1]*(2-1)=[2,1]
        layers = []                                     # 생성된 block들을 순서대로 담을 리스트 초기화
        for stride in strides:                          # 각 block마다 stride 값을 적용하며 반복
            # residual block 생성 후 리스트에 추가
            layers.append(block(self.in_channels, out_channels, stride))
            # 다음 block을 위해 입력 채널 수 업데이트
            self.in_channels = out_channels * block.expansion   
        
        # 리스트에 담긴 block들을 순차적으로 연결하여 하나의 레이어로 반환
        return nn.Sequential(*layers)   # *: 리스트 언패킹 연산자, 리스트의 요소들을 개별 인자로 반환하여 nn.Sequential에 전달
    
    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)     # flatten
        output = self.fc(x)
        return output

    # ResNet 모델의 가중치 초기화 함수
    def _initialize_weights(self):
        for m in self.modules():     # 모델의 모든 모듈(레이어)을 순회
            if isinstance(m, nn.Conv2d):    # Conv2d 레이어인 경우
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # He 초기화 방법 적용
                if m.bias is not None:       # 편향이 존재하는 경우 편향 초기화
                    nn.init.constant_(m.bias, 0)   # 편향을 0으로 초기화
            elif isinstance(m, nn.BatchNorm2d):   # BatchNorm2d 레이어인 경우
                nn.init.constant_(m.weight, 1)     # 배치 정규화의 가중치를 1로 초기화
                nn.init.constant_(m.bias, 0)       # 배치 정규화의 편향을 0으로 초기화
            elif isinstance(m, nn.Linear):        # Linear 레이어인 경우
                nn.init.normal_(m.weight, 0, 0.01)   # 선형 레이어의 가중치를 평균 0, 표준편차 0.01인 정규분포로 초기화
                nn.init.constant_(m.bias, 0)           # 선형 레이어의 편향을 0으로 초기화
    
    # ResNet 정의
    def resnet18():
        return ResNet(BasicBlock, [2,2,2,2])
    
    def resnet34():
        return ResNet(BasicBlock, [3,4,6,3])
    
    def resnet50():
        return ResNet(BottleneckBlock, [3,4,6,3])
    
    def resnet101():
        return ResNet(BottleneckBlock, [3,4,23,3])
    
    def resnet152():
        return ResNet(BottleneckBlock, [3,8,36,3])
    

# Model Train
print("-" * 50)
print("Model Training: device =", device)
print("-" * 50)

# ResNet-50 모델 생성 후 지정한 장치(GPU 또는 CPU)로 이동
model = ResNet.resnet50().to(device)

# 랜덤 입력 데이터 생성 (배치 = 3, 채널 = 3, 이미지 크기 = 224x224) 후 장치로 이동
x = torch.randn(3, 3, 224, 224).to(device)

# 모델에 입력 데이터를 넣어 forward 연산 수행
output = model(x)
print("Output size:", output.size())    # 출력 텐서의 크기 출력 (보통 [batch_size, num_classes])

# 모델 구조 요약 출력 (입력 크기 기준으로 각 레이어 정보 확인)
# params: 모델, 입력 tensor shape(채널, 높이, 너비), 장치 지정)
summary(model, (3, 224, 224), device=device.type)

# reduction='sum' -> 배치 내 모든 샘플의 loss를 평균(mean)으로 계산
loss_func = nn.CrossEntropyLoss(reduction='mean')

# Adam 옵티마이저 설정
# model.parameters(): 모델의 학습 가능한 매개변수들을 반환하여 옵티마이저에 전달
# lr=0.001: 학습률 설정
opt = optim.Adam(model.parameters(), lr=0.001)

# 학습률 스케줄러 클래스 불러오기
# ReduceLROnPlateau: 검증 성능이 개선되지 않을 때 학습률을 감소시키는 스케줄러
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 학습률 스케줄러 설정
lr_scheduler = ReduceLROnPlateau(opt,           # 적용할 옵티마이저
                                mode='min',     # 모니터링 값이 "감소"할 때 개선으로 판단
                                factor=0.1,     # 학습률을 0.1배로 감소 (예: 0.001 -> 0.0001)
                                patience=10)    # patience: 개선이 없는 에폭 수 - 10 epoch 동안 검증 성능이 개선되지 않으면 학습률 감소

# function to get current lr
# 옵티마이저의 현재 학습률을 반환하는 함수
def get_lr(opt):
    # PyTorch 옵티마이저는 param_groups라는 리스트로 파라미터 그룹과 학습률(lr) 관리
    # 보통 하나의 그룹만 쓰지만, 여러 그룹 가능

    for param_group in opt.param_groups:
        # 첫 번째 param_group의 'lr' 값을 반환
        return param_group['lr']
    
# function to calculate matric per mini-batch
# 한 배치(batch)에서 정확히 맞춘 샘플 수를 계산하는 함수
def metric_batch(output, target):   # params: 예측값, 정답
    pred = output.argmax(dim=1, keepdim=True)   # 예측값에서 가장 높은 확률을 가진 클래스 인덱스 추출 (예: [0, 2, 1])
                                                # dim=1: 클래스 차원 기준(32개의 예측값)
                                                # keepdim=True: 결과 텐서의 차원을 유지하여 (배치 크기, 1) 형태로 반환
    corrects = pred.eq(target.view_as(pred)).sum().item()   # 예측과 정답이 일치하는 샘플 수 계산
    return corrects                             # 배치 내 맞춘 샘플 수 반환

# function to calculate loss per mini-batch
# 한 배치(batch)에서 손실(loss) 계산과 metric(정확도) 계산,
# 옵티마이저가 주어지면 역전파와 파라미터 업데이트까지 수행하는 함수
def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)            # output과 target을 사용하여 손실 계산 (예: CrossEntropyLoss)
    metric_b = metric_batch(output, target)     # 한 배치의 정확한 예측 개수를 metric_batch 함수를 사용하여 계산

    if opt is not None:         # 옵티마이저가 주어졌다면 학습 단계 수행
        opt.zero_grad()     # 이전 그래디언트 초기화 (다음 배치에서 누적 방지)
        loss.backward()     # 손실에 대한 역전파 수행 (그래디언트 계산)
        opt.step()          # 역전파 수행, 옵티마이저를 사용하여 모델의 파라미터 업데이트
    
    return loss.item(), metric_b   # loss, 맞춘 샘플 수 반환

# function to calculate loss and metric per epoch
def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb = xb.to(device)   # 입력 배치를 장치로 이동 (GPU 또는 CPU)
        yb = yb.to(device)   # 레이블 배치를 장치로 이동
        output = model(xb)   # 모델에 입력 배치를 넣어 예측값 계산

        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)   # 배치 단위로 손실과 정확도 계산
        running_loss += loss_b * xb.size(0)   # 배치 손실에 배치 크기를 곱하여 누적 (전체 손실 계산을 위해)
        
        if metric_b is not None:   # metric_b가 None이 아닌 경우에만 누적
            running_metric += metric_b   # 배치 단위로 맞춘 샘플 수 누적

        if sanity_check:   # sanity_check가 True인 경우, 첫 번째 배치만 처리하고 루프 종료
            break

    loss = running_loss / len_data   # 전체 데이터 수로 나누어 평균 손실 계산
    metric = running_metric / len_data if metric_b is not None else None   # 전체 데이터 수로 나누어 평균 정확도 계산 (metric_b가 None이 아닌 경우에만 계산)

    return loss, metric

# function to start training
def train_val(model, params):
    num_epochs = params['num_epochs']
    loss_func = params['loss_func']
    opt = params['optimizer']
    train_dl = params['train_dl']
    val_dl = params['val_dl']
    sanity_check = params.get('sanity_check')   # 디버깅용 샘플만 돌릴지 여부
    lr_scheduler = params['lr_scheduler']       # 학습률 스케줄러
    path2weights = params.get('path2weights')     # 학습 완료 모델 저장 경로

    # 학습 및 검증 loss와 metric을 기록한 딕셔너리 초기화
    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    # validation loss 최소값 초기화 (비교용)
    best_loss = float('inf')        # 초기에는 무한대로 설정하여 첫 epoch에서 갱신 가능
    
    start_time = time.time()   # 전체 학습 시작 시간 기록

    for epoch in range(num_epochs):     # 현재 epoch 수 만큼 반복
        current_lr = get_lr(opt)   # 현재 optimizer 학습률 확인
        print('Epoch{}/{}, current lr={}'.format(epoch, num_epochs-1, current_lr))

        model.train()  # 모델을 학습 모드로 설정
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt=opt)
        loss_history['train'].append(train_loss)        # 학습 데이터로 loss, metric 계산
        metric_history['train'].append(train_metric)    # 학습 loss와 metric 기록

        model.eval()   # 모델을 평가 모드로 설정
        with torch.no_grad():   # 평가 단계에서는 그래디언트 계산 비활성화
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)   # 검증 데이터로 손실과 정확도 계산
        loss_history['val'].append(val_loss)        # 검증 데이터로 loss, metric 계산
        metric_history['val'].append(val_metric)    # 검증 loss와 metric 기록

        if val_loss < best_loss:   # 이전까지의 최소 validation loss보다 낮으면 best_loss 갱신
            best_loss = val_loss  
            torch.save(model.state_dict(), path2weights) 
            print('Get best model (loss: {:.4f})'.format(best_loss))

        lr_scheduler.step(val_loss)   # validation loss를 기준으로 학습률 조정 (ReduceLROnPlateau)
        print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time:%.4fmin'
                % (train_loss, val_loss, val_metric, (time.time() - start_time) / 60))
        print("-" * 50)

    return model, loss_history, metric_history

# test evaluation 함수 추가
def evaluate_test(model, test_dl, loss_func, class_names=None):
    # 모델을 평가 모드로 전환
    model.eval()

    # 기록할 데이터 초기화
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():   # 테스트 단계에서 gradient 계산 비활성화
        for xb, yb in test_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            # forward pass -> 모델 예측값 계산
            output = model(xb)
            # 배치 단위 loss 계산
            loss = loss_func(output,yb)
            # 배치 loss 누적
            running_loss += loss.item() * xb.size(0)
            # 가장 높은 확률을 가진 클래스 인덱스 선택
            preds = output.argmax(dim=1)
            # accuracy 계산
            running_corrects += preds.eq(yb).sum().item()

            # cpu로 이동 후 numpy로 변환하여 리스트 저장
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())
    total = len(test_dl.dataset)
    avg_loss = running_loss / total
    accuracy = running_corrects / total

    print("=" * 50)
    print("TEST EVALUATION RESULTS")
    print("=" * 50)
    print(f"Test Loss    : {avg_loss:.6f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    return avg_loss, accuracy, np.array(all_preds), np.array(all_targets)

# define the training parameters
params_train = {
    'num_epochs': 20,
    'optimizer': opt,
    'loss_func': loss_func,
    'train_dl': train_dl,
    'val_dl': val_dl,
    'sanity_check': False,
    'lr_scheduler': lr_scheduler,
    'path2weights': './models/weights/resnet50_baseline.pt'
}

# create the directory that stores weights.pt
# 지정한 경로(directory)에 폴더가 없으면 새로 생성하는 함수
def createFolder(directory):
    try:
        if not os.path.exists(directory):   # directory가 존재하지 않는 경우
            os.makedirs(directory)          # directory 생성
    except OSError as e:                   # OS 관련 오류가 발생한 경우 예외 처리
        print(f"Error creating directory {directory}: {e}")   # 오류 메시지 출력

createFolder('./models/weights')
model, loss_hist, metric_hist = train_val(model, params_train)

# Train-Validation Progress
num_epochs = params_train['num_epochs']

createFolder('./results/baseline')

# best validation model load
model.load_state_dict(torch.load(params_train['path2weights']))
model.eval()

# 클래스 이름 매핑
class_names = ['airplane', 'bird', 'car', 'cat', 'deer',
               'dog', 'horse', 'monkey', 'ship', 'truck']

# 테스트 함수 실행
test_loss, test_acc, test_preds, test_targets = evaluate_test(
    model, test_dl, loss_func, class_names=class_names
)

# 로그 저장
with open('./results/baseline/result_log.txt', 'w') as f:
    f.write(f"Best val loss: {min(loss_hist['val']):.6f}\n")
    f.write(f"Best val acc: {max(metric_hist['val']):.6f}\n")
    f.write(f"Test loss: {test_loss:.6f}\n")
    f.write(f"Test accuracy: {test_acc:.6f}\n")

# plot loss progress
plt.title("Train-Val loss")
plt.plot(range(1, num_epochs+1), loss_hist['train'], label='train')
plt.plot(range(1, num_epochs+1), loss_hist['val'], label='val')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig('./results/baseline/loss_progress.png')
plt.close()

# plot accuracy progress
plt.title("Train-Val Accuracy")
plt.plot(range(1, num_epochs+1), metric_hist['train'], label='train')
plt.plot(range(1, num_epochs+1), metric_hist['val'], label='val')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig('./results/baseline/accuracy_progress.png')
plt.close()