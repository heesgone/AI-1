# 과제 2. ImageFolder 적용이 가능한 새로운 데이터셋을 찾아서 Dataset Loading을 구현하고 CNN을 적용하여 정확도를 향상시켜보세요.
# 보고서 작성 - 데이터셋 설명, 기본 CNN, 성능 향상 방법
# Epoch 증가 (10-> 20 -> 30)

# 경로 지정
import os
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

#########################################
# CNN Model
#########################################
print("\n==== CNN 모델 설계 시작 ====")

import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()   
        
        self.first_pass = True      # 처음 한 번만 출력하기 위한 플래그 설정

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1)  
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.fc1 = nn.Linear(128*32*32, 512)
        self.fc2 = nn.Linear(512, 5)

    def forward(self, x):
        # self.first_pass가 True일 때만 아래 블록 실행
        if self.first_pass:
            print("\n--- 데이터 흐름 확인 ---")
            print("입력: ", x.size())
            
            x = F.relu(self.conv1(x))
            print("conv1 연산 후: ", x.size())
            
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            print("pooling 연산 후(커널 2, 스트라이드 2): ", x.size())
            
            x = F.relu(self.conv2(x))
            print("conv2 연산 후: ", x.size())
            
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            print("pooling 연산 후(커널 2, 스트라이드 2): ", x.size())

            x = F.relu(self.conv3(x))
            print("conv3 연산 후: ", x.size())
            
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            print("pooling 연산 후(커널 2, 스트라이드 2): ", x.size())
            
            x = x.view(-1, 128*32*32)
            print("평탄화 후(차원 감소): ", x.size())
            
            x = F.relu(self.fc1(x))
            print("fc1 연산 후: ", x.size())
            
            x = self.fc2(x)
            print("fc2 연산 후(결과): ", x.size())
            print("--- 확인 완료! 이후 출력은 생략합니다 ---\n")
            
            # 출력이 끝났으니 플래그를 False로 변경
            self.first_pass = False
            
        else:
            # 두 번째 배치부터는 출력 없이 연산만 수행
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = x.view(-1, 128*32*32)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

        return x
    
cnn = CNN().to(device) # 객체 생성

#########################################
# Parameters
#########################################
print("\n==== CNN 객체 생성 완료, 파라미터 설정 ====")
import torch.optim as optim
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.01)

#########################################
# Train
#########################################
print("\n==== Train 시작 ====")

train_losses = []
val_losses = []
val_accs = []

for epoch in range(30): # epoch 30으로 증가
    cnn.train()
    train_loss = 0
    for index, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = cnn(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    cnn.eval() # 검증을 위해 모델을 평가 모드로 전환
    val_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            output = cnn(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # 1 epoch마다 평균 loss와 정확도 출력
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / len(val_dataset)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    val_accs.append(val_accuracy)

    print("[Epoch {}/30] Train Loss: {:.4f} | Val Loss: {:.4f} | Val Acc: {:.2f}%".format(
        epoch+1, avg_train_loss, avg_val_loss, val_accuracy
    ))

print("==== Train 완료 ====")

#########################################
# 학습 그래프 저장 (train_loss, val_loss)
#########################################
import matplotlib.pyplot as plt
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
plt.savefig('training_curves_dev2(epoch=30).png')
print("\n=== 학습 그래프(training_curves_dev2(epoch=30).png) 저장 완료 ===")



#########################################
# Test
#########################################
print("\n==== Test 시작 ====")

cnn.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = cnn(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} {:.0f}%\n".format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)
))
