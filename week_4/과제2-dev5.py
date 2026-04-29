# 과제 2. ImageFolder 적용이 가능한 새로운 데이터셋을 찾아서 Dataset Loading을 구현하고 CNN을 적용하여 정확도를 향상시켜보세요.
# 보고서 작성 - 데이터셋 설명, 기본 CNN, 성능 향상 방법
# 과적합 완화 기법 비교 (드롭아웃(0.5), l2 규제(1e-4), 둘다 적용)

# 경로 지정
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
root = './'
image_folder_path = os.path.join(root, 'images')

if not os.path.exists(image_folder_path):
    os.makedirs(image_folder_path)

import glob
print(image_folder_path)
print(glob.glob(image_folder_path+'/food11/*'))    # 해당 폴더 안의 모든 하위 폴더 목록 출력

#########################################
# Dataset Image Display
#########################################
# foold11/train/sushi 폴더 하위에 위치한 .png 파일 10개 출력
glob.glob(image_folder_path+'/food11/train/sushi/*')[:10]     # [:10] = [0:10:1] = 0부터 9까지 step 1씩 이동

#########################################
# Device setting
#########################################
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device=", device)

#########################################
# Image Folder
#########################################
from torchvision import datasets, transforms

# 1. 전처리 과정에 Resize 추가 (서로 다른 음식 사진 크기를 256x256으로 통일)
# 이미지 크기가 커질수록 학습이 정교해지지만, 256 정도가 속도와 성능 면에서 적당한 크기
transform = transforms.Compose([
    transforms.Resize((256, 256)), # 괄호가 두 개인 것 주의! ((H, W))
    transforms.ToTensor()
])

train_folder = datasets.ImageFolder(root=image_folder_path+'/food11/train',
                                    transform = transform)
class_names = train_folder.classes

test_folder = datasets.ImageFolder(root=image_folder_path+'/food11/test',
                                    transform = transform)

print(train_folder.class_to_idx)

#########################################
# Data Loader
#########################################
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
batch_size = 32

# train_folder에 있는 모든 데이터의 인덱스아 라벨을 가져와 리스트로 저장
targets = train_folder.targets
indices = list(range(len(train_folder)))

# trian_test_split의 'stratify' 매개변수에 targets를 전달하여 클래스 비율을 유지하면서 데이터를 분할
train_indices, val_indices = train_test_split(
    indices, 
    test_size=0.2, 
    random_state=42, 
    stratify=targets
)

# Subset을 사용하여 train_folder에서 train_indices와 val_indices에 해당하는 데이터만 추출하여 새로운 Dataset 객체 생성
train_dataset = Subset(train_folder, train_indices)
val_dataset = Subset(train_folder, val_indices)


total_size = len(train_folder)+len(test_folder)      # 전체 데이터셋 길이
print("\n==== Data Split 결과 ====")
print("전체 데이터 길이: ", total_size)
print("전체 학습 데이터: ", len(train_folder))


# 기존과 동일하게 DataLoader 생성
train_loader = torch.utils.data.DataLoader(train_dataset,           
                                           batch_size=batch_size,   
                                           shuffle=True,            
                                           num_workers=8)           
print("train dataset 길이: ", len(train_dataset))

val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,           
                                            num_workers=8)
print("validation dataset 길이: ", len(val_dataset))

test_loader = torch.utils.data.DataLoader(test_folder,             
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=8)
print("test dataset 길이: ", len(test_folder))


# images, labels에 각각 1 batch만큼 로드
images, labels = next(iter(train_loader))
print("1 batch당 train dataset 안의 image 행렬 크기: ", images.shape)
print("1 batch당 train dataset 안의 label 행렬 크기: ", labels.shape)


# 피처 맵 저장 함수
def save_feature_maps(feature_map, layer_name, exp_name, actual_label, num_kernels=16):
    f_map = feature_map.squeeze(0).cpu() # [C, H, W] 형태로 변환
    
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f'[{exp_name}] Feature Maps of {layer_name} (Target: {actual_label})', fontsize=16)
    
    # 앞의 16개 커널 결과만 시각화
    for i in range(num_kernels):
        plt.subplot(4, 4, i+1)
        plt.imshow(f_map[i], cmap='viridis')
        plt.axis('off')
        plt.title(f'Kernel {i}')
    
    plt.tight_layout()
    filename = f'feature_map_{layer_name}_dev5_{exp_name}.png'
    plt.savefig(filename)
    plt.close() 
    print(f"--- {layer_name} 피처 맵 저장 완료 ({filename}) ---")


#########################################
# CNN Model
#########################################
print("\n==== CNN 모델 설계 시작 ====")

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class CNN(nn.Module):
    def __init__(self, use_dropout=False):
        super(CNN, self).__init__()   
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1)  
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128*32*32, 512)
        self.dropout = nn.Dropout(0.5) # 드롭아웃 레이어 추가
        self.fc2 = nn.Linear(512, 5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 128*32*32)
        x = F.relu(self.fc1(x))

        if self.use_dropout:
            x = self.dropout(x) # 드롭아웃 적용

        x = self.fc2(x)    

        return x

# 실험 설정
experiments = [
    {"name": "Dropout_Only", "dropout": True, "l2": 0.0},
    {"name": "L2_Only", "dropout": False, "l2": 1e-4},
    {"name": "Dropout_Plus_L2", "dropout": True, "l2": 1e-4}
]

# 실험 반복 루프
for exp in experiments:
    print(f"\n==== Experiment: {exp['name']} ====")

    model = CNN(use_dropout=exp['dropout']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=exp['l2'])
    criterion = torch.nn.CrossEntropyLoss()

    train_losses, val_losses, val_accs = [], [], []
    print("\n==== Train 시작 ====")
    for epoch in range(20):
        model.train()
        train_loss = 0
        for index, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval() # 검증을 위해 모델을 평가 모드로 전환
        val_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)

                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * correct / len(val_dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accs.append(val_accuracy)

        print("[Epoch {}/20] Train Loss: {:.4f} | Val Loss: {:.4f} | Val Acc: {:.2f}%".format(
            epoch+1, avg_train_loss, avg_val_loss, val_accuracy
        ))
    print("==== Train 완료 ====")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 3, 2)
    plt.plot(val_accs, label='Val Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'training_curves_dev5_{exp["name"]}.png')
    print(f"\n=== 학습 그래프(training_curves_dev5_{exp['name']}.png) 저장 완료 ===")

    print("\n==== Test 시작 ====")
    all_preds, all_targets = [], []
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            test_loss += criterion(output, target).item()
            correct += pred.eq(target.view_as(pred)).sum().item()
            print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} {:.0f}%\n".format(
                    test_loss, correct, len(test_loader.dataset),
                    100. * correct / len(test_loader.dataset)
                ))
        

    # 혼동 행렬 저장
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title(f"Confusion Matrix: {exp['name']}")
    plt.savefig(f"cm_{exp['name']}.png")
    plt.close()

    print(f"\n📊 {exp['name']} 클래스별 리포트:")
    print(classification_report(all_targets, all_preds, target_names=class_names))

    # --- 실험 루프 마지막 부분에 피처 맵 시각화 추가 ---
    print(f"\n==== [{exp['name']}] 피처 맵 시각화 시작 ====")
    
    # 1. 샘플 이미지 준비
    sample_images, sample_labels = next(iter(test_loader))
    sample_img = sample_images[0].unsqueeze(0).to(device)
    actual_label = class_names[sample_labels[0]]

    # 2. 레이어 통과 (현재 루프의 model 사용)
    model.eval()
    with torch.no_grad():
        # 첫 번째 Conv 레이어
        f_map1 = F.relu(model.bn1(model.conv1(sample_img)))
        
        # 두 번째 Conv 레이어
        x = F.max_pool2d(f_map1, 2)
        f_map2 = F.relu(model.bn2(model.conv2(x)))

        # 세 번째 Conv 레이어
        x = F.max_pool2d(f_map2, 2)
        f_map3 = F.relu(model.bn3(model.conv3(x)))

    # 3. 함수 호출 (실험 이름 전달)
    save_feature_maps(f_map1, 'conv1', exp['name'], actual_label)
    save_feature_maps(f_map2, 'conv2', exp['name'], actual_label)
    save_feature_maps(f_map3, 'conv3', exp['name'], actual_label)

    print(f"==== [{exp['name']}] 모든 과정 완료 ====\n")