# 과제 1. Rock-Paper-Scissors Dataset에 적용한 ImageFolder 코드에서 train data, validation data로 split하는 코드를 추가
# Data split 결과 출력 (Train data 개수, Validation data 개수)
# 보고서 작성 - Data Split 코드 설명(line by line 상세하게 설명)

# 경로 지정
import os
root = './'
image_folder_path = os.path.join(root, 'images')

# images 폴더가 없으면 생성
if not os.path.exists(image_folder_path):
    os.makedirs(image_folder_path)

import urllib.request
url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
local_zip = os.path.join(root, 'rps.zip')
if not os.path.exists(local_zip):
    print("데이터 다운로드 중")
    urllib.request.urlretrieve(url, local_zip)
else:
    print("이미 rps.zip 파일이 존재합니다.")

# images 폴더에 압축해제
import zipfile
with zipfile.ZipFile(local_zip, 'r') as zip_ref:    # zip 파일을 read 용도로 열어서
    zip_ref.extractall(image_folder_path)           # image_folder_path 폴더에 압축해제

import glob
print(image_folder_path)
print(glob.glob(image_folder_path+'/rps/*'))    # 해당 폴더 안의 모든 하위 폴더 목록 출력

#########################################
# Dataset Image Display
#########################################
# rock 폴더 하위에 위치한 .png 파일 10개 출력
# glob: 파일 시스템에서 특정 패턴과 일치하는 파일들을 찾는 데 사용되는 라이브러리
glob.glob(image_folder_path+'/rps/rock/*')[:10]     # [:10] = [0:10:1] = 0부터 9까지 step 1씩 이동

#########################################
# Image Folder
#########################################
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device=", device)

from torchvision import datasets, transforms
image_folder = datasets.ImageFolder(root=image_folder_path+'/rps',
                                    transform = transforms.Compose([
                                        transforms.ToTensor()
                                    ]))

# class to index 라벨값 확인 (추후 시각화에 사용)
print(image_folder.class_to_idx)

#########################################
# Data Loader
#########################################
from torch.utils.data import Dataset, random_split
batch_size = 32

total_size = len(image_folder)      # 전체 데이터셋 길이
print("\n==== Data Split 결과 ====")
print("전체 데이터 길이: ", total_size)

# train:val:test=6.4:1.6:2로 설정
test_size = int(total_size*0.2)
train_size = int(total_size*0.64)    
val_size = total_size - train_size - test_size

# random_split: Dataset을 train과 validation, test로 분할
train_dataset, val_dataset, test_dataset = random_split(image_folder, [train_size, val_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset,            # image_folder 지정
                                           batch_size=batch_size,   # 배치 사이즈 가정
                                           shuffle=True,            # shuffle 여부 지정
                                           num_workers=8)           # 데이터 로딩을 병렬로
print("train dataset 길이: ", len(train_dataset))

val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,           # 검증 데이터는 굳이 섞을 필요 없습니다 (False)
                                            num_workers=8)
print("validation dataset 길이: ", len(val_dataset))

test_loader = torch.utils.data.DataLoader(test_dataset,             # train 데이터와 같은 폴더에서 가져옴 (대신 셔플 안 된 상태)
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=8)
print("test dataset 길이: ", len(test_dataset))


# images, labels에 각각 batch가 로드
images, labels = next(iter(train_loader))   # 1개 batch 추출
print("1 batch당 train dataset 안의 image 행렬 크기: ", images.shape)
print("1 batch당 train dataset 안의 label 행렬 크기: ", labels.shape)

#########################################
# Train Image Display
#########################################
import matplotlib.pyplot as plt

labels_map = {v:k for k, v in image_folder.class_to_idx.items()}

figure = plt.figure(figsize=(12,8))
cols, rows = 8, 5

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(images), size=(1,)).item()
    img, label = images[sample_idx], labels[sample_idx].item()
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(torch.permute(img, (1, 2, 0)))
plt.show()


print("\n==== CNN 모델 설계 시작 ====")

import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()   
        
        self.first_pass = True      # 처음 한 번만 출력하기 위한 플래그 설정

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1)  
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.fc1 = nn.Linear(64*75*75, 512)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        # self.first_pass가 True일 때만 아래 블록 실행
        if self.first_pass:
            print("\n--- 첫 번째 배치 데이터 흐름 확인 ---")
            print("입력: ", x.size())
            
            x = F.relu(self.conv1(x))
            print("conv1 연산 후: ", x.size())
            
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            print("pooling 연산 후(커널 2, 스트라이드 2): ", x.size())
            
            x = F.relu(self.conv2(x))
            print("conv2 연산 후: ", x.size())
            
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            print("pooling 연산 후(커널 2, 스트라이드 2): ", x.size())
            
            x = x.view(-1, 64*75*75)
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
            x = x.view(-1, 64*75*75)
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

for epoch in range(10):
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

    print("[Epoch {}/10] Train Loss: {:.4f} | Val Loss: {:.4f} | Val Acc: {:.2f}%".format(
        epoch+1, avg_train_loss, avg_val_loss, val_accuracy
    ))

print("==== Train 완료 ====")
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

print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} (:.0f)%\n".format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)
))