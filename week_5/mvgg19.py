# MVGG19 논문 재현 실험
# "Industrial object, machine part and defect recognition towards fully automated
#  industrial monitoring employing deep learning. The case of multilevel VGG19"
# - Casting dataset (Kaggle) 재현
#
# [논문과의 재현 가정 - 보고서 명시 필요]
# - augmentation 세부 파라미터(rotation=10, translate=0.1, noise std=0.03): 논문 미명시
# - input size: 300x300 원본 해상도 유지 (Casting dataset 원본 크기 기준)
# - epoch 수, batch size: 논문 미명시 ("adjusted for each dataset"만 언급)
# - optimizer: Adam default parameters (논문 명시)
# - 10-fold CV: 논문 명시
# - fine-tuning: pretrained VGG19 backbone freeze,
#   BD layers + classifier trainable (논문 설명 기반 재현)
# - Dense 2500 + BN + 50% Dropout → Softmax: 논문 명시
# - sequential output = block5 이후 GAP feature로 구현 (논문 Figure 1 기반 재현 가정)

#########################################
# 재현성 seed 고정
#########################################
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

#########################################
# 라이브러리
#########################################
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device =", device)

train_dir = './images/casting_data/train'
test_dir  = './images/casting_data/test'

for d in ['./results/examples', './results/graphs', './results/matrix',
          './results/featuremaps', './results/checkpoints']:
    os.makedirs(d, exist_ok=True)


#########################################
# Dataset Image Display
#########################################
import glob
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

example_dir = './results/examples'
ok_dir  = os.path.join(train_dir, 'ok_front')
def_dir = os.path.join(train_dir, 'def_front')

ok_paths  = glob.glob(os.path.join(ok_dir,  '*'))
def_paths = glob.glob(os.path.join(def_dir, '*'))

ok_samples  = random.sample(ok_paths,  5)
def_samples = random.sample(def_paths, 5)

sample_paths  = ok_samples + def_samples
sample_labels = ['ok_front'] * 5 + ['def_front'] * 5

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()
for i, (img_path, label) in enumerate(zip(sample_paths, sample_labels)):
    img = Image.open(img_path).convert('RGB')
    axes[i].imshow(img)
    axes[i].set_title(label, fontsize=10)
    axes[i].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(example_dir, 'train_samples_5_ok_5_def.png'),
            bbox_inches='tight', dpi=200)
plt.close()
print("샘플 이미지 저장 완료")


#########################################
# Train / Test Class Distribution
#########################################
graph_dir = './results/graphs'

def analyze_class_distribution(data_dir, split_name, save_dir):
    class_counts = {}
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            class_counts[class_name] = len(glob.glob(os.path.join(class_path, '*')))
    total_count = sum(class_counts.values())
    print(f"\n[{split_name}] 클래스별 이미지 개수")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}장 ({count/total_count*100:.2f}%)")
    classes = list(class_counts.keys())
    counts  = list(class_counts.values())
    plt.figure(figsize=(6, 4))
    plt.bar(classes, counts)
    plt.title(f'Class Distribution ({split_name})')
    plt.xlabel('Class'); plt.ylabel('Number of Images')
    for i, count in enumerate(counts):
        plt.text(i, count + max(counts)*0.01, str(count), ha='center')
    plt.savefig(os.path.join(save_dir, f'class_distribution_{split_name.lower()}.png'),
                bbox_inches='tight')
    plt.close()
    return class_counts

train_counts = analyze_class_distribution(train_dir, 'Train', graph_dir)
test_counts  = analyze_class_distribution(test_dir,  'Test',  graph_dir)


#########################################
# Transform 정의
# 논문 명시 augmentation: Gaussian Noise, Rotation, H/V Flip, Height/Width Shift
# pretrained VGG19 fine-tuning → ImageNet 기준 Normalize
#########################################
from torchvision import datasets, transforms, models

class AddGaussianNoise(object):
    """
    논문 명시 augmentation
    clamp(0,1)로 노이즈 적용 후 픽셀값 범위 보장
    """
    def __init__(self, mean=0., std=0.03):
        self.mean = mean
        self.std  = std
    def __call__(self, tensor):
        noisy = tensor + torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(noisy, 0.0, 1.0)

# pretrained VGG19 사용 → ImageNet 기준 Normalize
IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

train_transform = transforms.Compose([                          
    transforms.Grayscale(num_output_channels=3),            # 흑백 → 3채널
    transforms.RandomRotation(10),                          # 논문 명시
    transforms.RandomHorizontalFlip(),                      # 논문 명시
    transforms.RandomVerticalFlip(),                        # 논문 명시
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 논문: height/width shift
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.03),                             # 논문 명시
    IMAGENET_NORMALIZE,
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    IMAGENET_NORMALIZE,
])


#########################################
# ImageFolder + ConcatDataset
# train/test 합쳐서 전체 데이터 → 10-fold CV (논문 명시)
#########################################
from torch.utils.data import DataLoader, ConcatDataset, Subset

train_folder_train_tf = datasets.ImageFolder(train_dir, transform=train_transform)
test_folder_train_tf  = datasets.ImageFolder(test_dir,  transform=train_transform)
full_dataset_train_tf = ConcatDataset([train_folder_train_tf, test_folder_train_tf])

train_folder_val_tf   = datasets.ImageFolder(train_dir, transform=val_transform)
test_folder_val_tf    = datasets.ImageFolder(test_dir,  transform=val_transform)
full_dataset_val_tf   = ConcatDataset([train_folder_val_tf, test_folder_val_tf])

class_names = train_folder_train_tf.classes
print(f"\n클래스 인덱스: {train_folder_train_tf.class_to_idx}")
# def_front=0, ok_front=1 (알파벳 순서 기준)

DEFECT_LABEL = train_folder_train_tf.class_to_idx['def_front']  # = 0
print(f"Positive class (결함) = '{class_names[DEFECT_LABEL]}' (label={DEFECT_LABEL})")

all_targets = np.array(
    train_folder_train_tf.targets + test_folder_train_tf.targets
)
print(f"전체 데이터 수: {len(all_targets)}")


#########################################
# MVGG19 Model
#
# 논문 Figure 1 재현:
#   - pretrained VGG19 backbone (전체 freeze → fine-tuning)
#   - 2nd/3rd/4th MaxPool 이후 BD layer (BN + Dropout50% + GlobalAvgPool)
#   - 3개 BD output + block5 GAP output → concatenation (1408-dim)
#   - Dense 2500 + BN + Dropout50% → num_classes
#
# 재현 가정:
#   - sequential output = block5 이후 GAP feature (논문 Figure 1 기반)
#########################################
import torch.nn as nn

class BDLayer(nn.Module):
    """
    BD = Batch Normalization + Dropout(50%) + Global Average Pooling
    논문 Figure 1: 2nd/3rd/4th MaxPool 이후에 붙는 보조 경로
    """
    def __init__(self, in_channels):
        super(BDLayer, self).__init__()
        self.bn      = nn.BatchNorm2d(in_channels)
        self.dropout = nn.Dropout2d(p=0.5)
        self.gap     = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.bn(x)
        x = self.dropout(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x


class MVGG19(nn.Module):
    """
    Multipath VGG19 (논문 Figure 1 재현)

    BD 경로:
      Block2 MaxPool 이후 → BD (128-dim)
      Block3 MaxPool 이후 → BD (256-dim)
      Block4 MaxPool 이후 → BD (512-dim)
    Sequential 경로:
      Block5 MaxPool 이후 → GAP (512-dim)
    → concat 1408-dim → Dense(2500) + BN + Dropout(50%) → num_classes
    """
    def __init__(self, num_classes=2):
        super(MVGG19, self).__init__()

        # pretrained VGG19 backbone
        backbone = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        features = backbone.features

        # 블록별 분리
        # Block1: [0:5]   conv×2 + maxpool  →  64ch, 150x150
        # Block2: [5:10]  conv×2 + maxpool  → 128ch,  75x75
        # Block3: [10:19] conv×4 + maxpool  → 256ch,  37x37
        # Block4: [19:28] conv×4 + maxpool  → 512ch,  18x18
        # Block5: [28:37] conv×4 + maxpool  → 512ch,   9x9
        self.block1 = features[0:5]
        self.block2 = features[5:10]
        self.block3 = features[10:19]
        self.block4 = features[19:28]
        self.block5 = features[28:37]

        # BD layers (논문 명시: 2nd/3rd/4th maxpool 이후)
        self.bd2 = BDLayer(128)
        self.bd3 = BDLayer(256)
        self.bd4 = BDLayer(512)

        # Sequential 경로: block5 이후 GAP
        self.gap5 = nn.AdaptiveAvgPool2d(1)

        # concat 크기: 128 + 256 + 512 + 512 = 1408
        concat_size = 128 + 256 + 512 + 512

        # Dense 2500 + BN + Dropout50% (논문 명시)
        self.classifier = nn.Sequential(
            nn.Linear(concat_size, 2500),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2500),
            nn.Dropout(p=0.5),
            nn.Linear(2500, num_classes),
        )

        # fine-tuning: backbone 전체 freeze (논문 명시)
        for block in [self.block1, self.block2, self.block3,
                      self.block4, self.block5]:
            for param in block.parameters():
                param.requires_grad = False
        # BD layers, classifier는 trainable

    def forward(self, x):
        x = self.block1(x)

        x = self.block2(x)
        bd2_out = self.bd2(x)                          # 128-dim

        x = self.block3(x)
        bd3_out = self.bd3(x)                          # 256-dim

        x = self.block4(x)
        bd4_out = self.bd4(x)                          # 512-dim

        x = self.block5(x)
        gap5_out = self.gap5(x).view(x.size(0), -1)   # 512-dim

        # concatenation → Dense 2500
        fused = torch.cat([bd2_out, bd3_out, bd4_out, gap5_out], dim=1)
        return self.classifier(fused)


#########################################
# 10-Fold Cross Validation
#########################################
print("\n==== MVGG19 10-Fold CV 시작 ====")
print("※ 재현 가정: input=300x300, epochs=10, batch=32, lr=0.001(Adam default)")
print("※ augmentation 파라미터(rotation=10, translate=0.1, noise std=0.03): 논문 미명시\n")

from sklearn.model_selection import StratifiedKFold
import torch.optim as optim
from sklearn.metrics import (confusion_matrix,
                              precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve)
import seaborn as sns

NUM_EPOCHS = 10    
BATCH_SIZE = 32    
LR         = 0.001 # Adam default (논문 명시)
N_SPLITS   = 10

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

fold_accs, fold_precisions, fold_recalls = [], [], []
fold_f1s, fold_aucs = [], []
all_fold_train_losses, all_fold_val_losses = [], []
all_fold_train_accs,   all_fold_val_accs   = [], []

for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(all_targets)), all_targets), 1):

    set_seed(42)  # fold별 재현성 고정

    print(f"\n{'='*45}")
    print(f"  Fold {fold}/{N_SPLITS}")
    print(f"{'='*45}")

    train_dataset = Subset(full_dataset_train_tf, train_idx)
    val_dataset   = Subset(full_dataset_val_tf,   val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4)
    print(f"  Train: {len(train_dataset)}장  |  Val: {len(val_dataset)}장")

    # fold마다 모델 새로 초기화
    model     = MVGG19(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    # trainable 파라미터만 optimizer에 전달 (frozen backbone 제외)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )

    fold_train_losses, fold_val_losses = [], []
    fold_train_accs,   fold_val_accs   = [], []

    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)
        train_loss /= len(train_loader)
        train_acc   = 100.0 * correct / total

        # Validation 
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total   += labels.size(0)
        val_loss /= len(val_loader)
        val_acc   = 100.0 * correct / total

        fold_train_losses.append(train_loss)
        fold_val_losses.append(val_loss)
        fold_train_accs.append(train_acc)
        fold_val_accs.append(val_acc)

        print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  "
              f"Val Acc: {val_acc:.2f}%")

    all_fold_train_losses.append(fold_train_losses)
    all_fold_val_losses.append(fold_val_losses)
    all_fold_train_accs.append(fold_train_accs)
    all_fold_val_accs.append(fold_val_accs)

    # 마지막 epoch까지 학습 완료 후 Test
    model.eval()

    all_preds = []
    all_labels_list = []
    all_probs = []

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:   # fold의 hidden set
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            probs_defect = torch.softmax(outputs, dim=1)[:, DEFECT_LABEL]
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())
            all_probs.extend(probs_defect.cpu().numpy())

    fold_loss = running_loss / len(val_loader)
    fold_acc = 100.0 * correct / total
    fold_prec = precision_score(all_labels_list, all_preds,
                                pos_label=DEFECT_LABEL, zero_division=0)
    fold_rec  = recall_score(all_labels_list, all_preds,
                             pos_label=DEFECT_LABEL, zero_division=0)
    fold_f1   = f1_score(all_labels_list, all_preds,
                         pos_label=DEFECT_LABEL, zero_division=0)
    binary_labels = [1 if l == DEFECT_LABEL else 0 for l in all_labels_list]
    fold_auc  = roc_auc_score(binary_labels, all_probs)

    fold_accs.append(fold_acc)
    fold_precisions.append(fold_prec)
    fold_recalls.append(fold_rec)
    fold_f1s.append(fold_f1)
    fold_aucs.append(fold_auc)

    print(f"\n  [Fold {fold} Result]  (Positive = def_front)")
    print(f"  Accuracy : {fold_acc:.2f}%")
    print(f"  Precision: {fold_prec:.4f}  Recall: {fold_rec:.4f}  "
          f"F1: {fold_f1:.4f}  AUC: {fold_auc:.4f}")

    # confusion matrix 저장
    cm = confusion_matrix(all_labels_list, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label'); plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - MVGG19 Fold {fold}')
    plt.savefig(f'./results/matrix/confusion_matrix_mvgg19_fold{fold}.png',
                bbox_inches='tight')
    plt.close()

    # fold별 loss / accuracy 곡선 저장
    epochs_range = range(1, NUM_EPOCHS + 1)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(epochs_range, fold_train_losses, label='Train Loss')
    axs[0].plot(epochs_range, fold_val_losses,   label='Val Loss')
    axs[0].set_title(f'Loss Curve - MVGG19 Fold {fold}')
    axs[0].set_xlabel('Epoch'); axs[0].set_ylabel('Loss')
    axs[0].legend(); axs[0].grid(True)
    axs[1].plot(epochs_range, fold_train_accs, label='Train Accuracy')
    axs[1].plot(epochs_range, fold_val_accs,   label='Val Accuracy')
    axs[1].set_title(f'Accuracy Curve - MVGG19 Fold {fold}')
    axs[1].set_xlabel('Epoch'); axs[1].set_ylabel('Accuracy (%)')
    axs[1].legend(); axs[1].grid(True)
    plt.tight_layout()
    plt.savefig(f'./results/graphs/curve_mvgg19_fold{fold}.png', bbox_inches='tight')
    plt.close()
    # =====================================================================
    # 논문 Figure 8 스타일 재현 (2x2 서브플롯)
    # - Accuracy / Loss는 앞 8 epoch만 사용
    # =====================================================================
    
    # 1. 2x2 그래프 틀 생성
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    bg_color = '#EAEAEA'

    # 2. 논문처럼 0~7 epoch만 시각화
    vis_epochs = range(0, 8)
    train_acc_vis  = fold_train_accs[:8]
    val_acc_vis    = fold_val_accs[:8]
    train_loss_vis = fold_train_losses[:8]
    val_loss_vis   = fold_val_losses[:8]

    # 3. ROC용 데이터
    y_true_binary = np.array(all_labels_list)
    y_scores_def = np.array(all_probs)          # defect 확률
    y_scores_ok  = 1.0 - y_scores_def           # ok 확률


    # a) Accuracy
    axs[0, 0].plot(vis_epochs, train_acc_vis, label='train', color='#C8524B', linewidth=1.5)
    axs[0, 0].plot(vis_epochs, val_acc_vis, label='validation', color='#4A76A8', linewidth=1.5)
    axs[0, 0].set_title('a)', loc='left', fontsize=16, fontweight='bold')
    axs[0, 0].set_title('Accuracy', fontsize=14)
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Accuracy (%)')
    axs[0, 0].legend(loc='upper left')
    axs[0, 0].set_facecolor(bg_color)
    axs[0, 0].grid(True, color='white', alpha=0.5)

    # b) Losses
    axs[0, 1].plot(vis_epochs, train_loss_vis, label='train', color='#C8524B', linewidth=1.5)
    axs[0, 1].plot(vis_epochs, val_loss_vis, label='validation', color='#4A76A8', linewidth=1.5)
    axs[0, 1].set_title('b)', loc='left', fontsize=16, fontweight='bold')
    axs[0, 1].set_title('Losses', fontsize=14)
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend(loc='upper left')
    axs[0, 1].set_facecolor(bg_color)
    axs[0, 1].grid(True, color='white', alpha=0.5)

    # c) ROC Curve
    fpr0, tpr0, _ = roc_curve((y_true_binary == DEFECT_LABEL).astype(int), y_scores_def)
    fpr1, tpr1, _ = roc_curve((y_true_binary != DEFECT_LABEL).astype(int), y_scores_ok)

    axs[1, 0].plot(fpr0, tpr0, label='class 0', color='#C8524B', linewidth=2)
    axs[1, 0].plot(fpr1, tpr1, label='class 1', color='#4A76A8', linewidth=2)
    axs[1, 0].set_title('c)', loc='left', fontsize=16, fontweight='bold')
    axs[1, 0].set_title('ROC curve', fontsize=14)
    axs[1, 0].set_xlabel('false positive rate')
    axs[1, 0].set_ylabel('true positive rate')
    axs[1, 0].legend(loc='lower right')
    axs[1, 0].set_facecolor(bg_color)
    axs[1, 0].grid(True, color='white', alpha=0.5)

    # d) Confusion Matrix
    cm = confusion_matrix(all_labels_list, all_preds)
    axs[1, 1].set_title('d)', loc='left', fontsize=16, fontweight='bold')
    axs[1, 1].set_title('Confusion matrix', fontsize=14)

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=True,
        xticklabels=['Defect', 'Ok'],
        yticklabels=['Defect', 'Ok'],
        ax=axs[1, 1]
    )
    axs[1, 1].set_xlabel('Predicted label')
    axs[1, 1].set_ylabel('True label')

    plt.tight_layout()
    plt.savefig(f'./results/graphs/figure8_replication_fold{fold}_mvgg19.png',
                bbox_inches='tight', dpi=200)
    plt.close()



#########################################
# 10-Fold 최종 결과 출력
#########################################
print(f"\n{'='*55}")
print(f"  MVGG19 - 10-Fold CV 최종 결과")
print(f"  (Positive class = def_front)")
print(f"{'='*55}")
print(f"  {'Metric':<12} {'Mean':>8}  {'Std':>8}")
print(f"  {'-'*32}")
print(f"  {'Accuracy':<12} {np.mean(fold_accs):>7.2f}%  ±{np.std(fold_accs):>5.2f}%")
print(f"  {'Precision':<12} {np.mean(fold_precisions):>8.4f}  ±{np.std(fold_precisions):>6.4f}")
print(f"  {'Recall':<12} {np.mean(fold_recalls):>8.4f}  ±{np.std(fold_recalls):>6.4f}")
print(f"  {'F1-score':<12} {np.mean(fold_f1s):>8.4f}  ±{np.std(fold_f1s):>6.4f}")
print(f"  {'AUC':<12} {np.mean(fold_aucs):>8.4f}  ±{np.std(fold_aucs):>6.4f}")
print(f"{'='*55}")

# 논문 결과와 비교 (Casting dataset, Table 1)
paper = {'acc': 77.62, 'prec': 77.68, 'rec': None, 'f1': None, 'auc': 94.94}
print(f"\n  [논문 결과 비교 - Casting dataset / Table 1]")
print(f"  {'Metric':<12} {'논문':>8}  {'재현':>8}  {'차이':>8}")
print(f"  {'-'*44}")
print(f"  {'Accuracy':<12} {paper['acc']:>7.2f}%  {np.mean(fold_accs):>7.2f}%  "
      f"{np.mean(fold_accs)-paper['acc']:>+7.2f}%")
print(f"  {'Precision':<12} {paper['prec']/100:>8.4f}  {np.mean(fold_precisions):>8.4f}  "
      f"{np.mean(fold_precisions)-paper['prec']/100:>+8.4f}")
print(f"  {'Recall':<12} {'  -':>8}  {np.mean(fold_recalls):>8.4f}  {'  -':>8}")
print(f"  {'F1-score':<12} {'  -':>8}  {np.mean(fold_f1s):>8.4f}  {'  -':>8}")
print(f"  {'AUC':<12} {paper['auc']/100:>8.4f}  {np.mean(fold_aucs):>8.4f}  "
      f"{np.mean(fold_aucs)-paper['auc']/100:>+8.4f}")


#########################################
# 10-Fold 요약 그래프 저장
#########################################
folds = list(range(1, N_SPLITS + 1))
metrics_dict = {
    'Accuracy (%)' : fold_accs,
    'Precision (%)'  : [v*100 for v in fold_precisions],
    'Recall (%)'     : [v*100 for v in fold_recalls],
    'F1-score (%)'   : [v*100 for v in fold_f1s],
    'AUC (%)'        : [v*100 for v in fold_aucs],
}

fig, axs = plt.subplots(2, 3, figsize=(16, 10))
axs = axs.flatten()
for i, (metric_name, values) in enumerate(metrics_dict.items()):
    axs[i].bar(folds, values, color='steelblue', alpha=0.8)
    axs[i].axhline(np.mean(values), color='red', linestyle='--',
                   label=f'Mean: {np.mean(values):.2f}')
    axs[i].set_title(metric_name)
    axs[i].set_xlabel('Fold'); axs[i].set_ylabel(metric_name)
    axs[i].set_xticks(folds); axs[i].legend(); axs[i].grid(True, alpha=0.3)

axs[5].axis('off')
summary_text = (
    "MVGG19  10-Fold CV Summary\n"
    "(Positive class = def_front)\n\n"
    f"Accuracy  : {np.mean(fold_accs):.2f}% ± {np.std(fold_accs):.2f}%\n"
    f"Precision : {np.mean(fold_precisions):.4f} ± {np.std(fold_precisions):.4f}\n"
    f"Recall    : {np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}\n"
    f"F1-score  : {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}\n"
    f"AUC       : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}\n\n"
    "--- 논문 결과 (Table 1) ---\n"
    f"Accuracy  : {paper['acc']:.2f}%\n"
    f"Precision : {paper['prec']/100:.4f}\n"
    f"AUC       : {paper['auc']/100:.4f}"
)
axs[5].text(0.05, 0.5, summary_text, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.suptitle('MVGG19 - 10-Fold CV Results (Casting Dataset)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('./results/graphs/10fold_summary_mvgg19.png', bbox_inches='tight', dpi=150)
plt.close()
print("\n요약 그래프 저장: ./results/graphs/10fold_summary_mvgg19.png")


#########################################
# Feature Map Visualization
# 마지막 fold의 마지막 epoch 모델 사용
#########################################
print("\n==== Feature Map 시각화 시작 ====")

sample_folder = datasets.ImageFolder(test_dir, transform=val_transform)
sample_loader = DataLoader(sample_folder, batch_size=32, shuffle=False)
sample_images, sample_labels_t = next(iter(sample_loader))
sample_img   = sample_images[0].unsqueeze(0).to(device)
actual_label = class_names[sample_labels_t[0].item()]

model.eval()

with torch.no_grad():
    x  = sample_img
    b1 = model.block1(x)
    b2 = model.block2(b1)
    b3 = model.block3(b2)
    b4 = model.block4(b3)
    b5 = model.block5(b4)

def save_feature_maps(block_out, block_name, label):
    fmaps = block_out.squeeze(0).cpu()
    plt.figure(figsize=(12, 12))
    for i in range(min(16, fmaps.shape[0])):
        plt.subplot(4, 4, i+1)
        plt.imshow(fmaps[i], cmap='viridis')
        plt.axis('off')
    plt.suptitle(f'{block_name} Feature Maps | Label: {label}')
    plt.tight_layout()
    plt.savefig(f'./results/featuremaps/{block_name.lower()}_featuremaps.png',
                bbox_inches='tight')
    plt.close()

save_feature_maps(b1, 'Block1', actual_label)
save_feature_maps(b2, 'Block2', actual_label)
save_feature_maps(b3, 'Block3', actual_label)
save_feature_maps(b4, 'Block4', actual_label)
save_feature_maps(b5, 'Block5', actual_label)

print("Feature Map 저장 완료: ./results/featuremaps/")
print("\n==== 전체 완료 ====")