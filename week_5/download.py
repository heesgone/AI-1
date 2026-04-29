import os
import shutil
import kagglehub

# 저장할 최종 경로
root = './images'
target_data_path = os.path.join(root, 'casting_data')
dst_train_dir = os.path.join(target_data_path, 'train')
dst_test_dir = os.path.join(target_data_path, 'test')

os.makedirs(target_data_path, exist_ok=True)

# 이미 정리되어 있으면 스킵
if os.path.exists(dst_train_dir) and os.path.exists(dst_test_dir):
    print("이미 데이터가 정리되어 있습니다.")
else:
    # 다운로드
    download_path = kagglehub.dataset_download(
        "ravirajsinh45/real-life-industrial-dataset-of-casting-product"
    )
    print("Downloaded dataset path:", download_path)

    # 원본 경로
    src_train_dir = os.path.join(download_path, 'casting_data', 'casting_data', 'train')
    src_test_dir = os.path.join(download_path, 'casting_data', 'casting_data', 'test')

    # 복사
    if not os.path.exists(dst_train_dir):
        shutil.copytree(src_train_dir, dst_train_dir)
        print("train 복사 완료")

    if not os.path.exists(dst_test_dir):
        shutil.copytree(src_test_dir, dst_test_dir)
        print("test 복사 완료")

    print("데이터 정리 완료")
    print("Final train path:", dst_train_dir)
    print("Final test path:", dst_test_dir)