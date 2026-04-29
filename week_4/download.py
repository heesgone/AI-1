import kagglehub


# 데이터셋 최신 버전 다운로드
path = kagglehub.dataset_download("imbikramsaha/food11")

print("데이터셋이 저장된 경로:", path)