
import numpy as np
import pandas as pd

class CFG:
    WINDOW_GIVEN = 100
    STRIDE = 60
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

def normalize_columns(data: np.ndarray) -> np.ndarray:
    """벡터화된 열 정규화 (z-정규화)"""
    means = data.mean(axis=0, keepdims=True)
    stds = data.std(axis=0, keepdims=True)
    
    # stds가 0인 경우 전체를 0으로 반환
    is_constant = (stds == 0)
    if np.any(is_constant):
        normalized_data = np.zeros_like(data)
        normalized_data[:, is_constant.squeeze()] = 0
    else:
        # z-정규화 수행
        normalized_data = (data - means) / stds
    return normalized_data

def prepare_training_data(df: pd.DataFrame, window_size: int, stride: int) -> np.ndarray:
    """학습 데이터 준비 - 윈도우 단위로 분할하고 정규화"""
    # 필요한 열 추출
    column_names = df.filter(regex='^P\d+$').columns.tolist()
    values = df[column_names].values.astype(np.float32)
    accident_labels = df.filter(regex='_flag$').values
    
    # 윈도우 시작 인덱스 계산
    potential_starts = np.arange(0, len(df) - window_size, stride)
    
    # 유효한 윈도우 필터링 (윈도우 마지막 다음 지점의 모든 flag가 0인 경우)
    valid_starts = [
        idx for idx in potential_starts 
        if (idx + window_size < len(df)) and (accident_labels[idx + window_size].sum() == 0)
    ]
    
    # 유효한 윈도우 추출
    windows = np.array([
        values[i:i + window_size] 
        for i in valid_starts
    ])  # Shape: (num_windows, window_size, num_features)
    
    # 각 윈도우 별로 정규화
    normalized_windows = np.array([
        normalize_columns(window) for window in windows
    ])  # 동일한 Shape 유지
    
    return normalized_windows, valid_starts  # Shape: (num_windows, window_size, num_features)

def get_labels(df: pd.DataFrame, valid_starts: list, window_size: int) -> np.ndarray:
    """유효한 윈도우의 레이블을 추출"""
    accident_labels = df.filter(regex='_flag$').values
    labels = np.array([accident_labels[idx + window_size] for idx in valid_starts])
    return labels

def merge_datasets(df_list):
    return pd.concat(df_list, ignore_index=True)