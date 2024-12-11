import numpy as np
import pandas as pd
import ast
from sklearn.metrics import f1_score

def calculate_pressure_level_f1(gt_df, pred_df):
    pred_flags = pred_df['flag_list'].apply(lambda x: np.array(ast.literal_eval(x)))
    gt_flags = gt_df['flag_list'].apply(lambda x: np.array(ast.literal_eval(x)))
    # 압력계 별 배점 가중치는 평가용으로 비공개
    # 단, 실제 답(GT)과 크게 다르지 않으며 정답에 근접한 예측에 대해 가산점을 주기 위한 용도.
    weight_flags = gt_df['weight_list'].apply(lambda x: np.array(ast.literal_eval(x)))

    total_f1 = 0
    valid_samples = 0

    for idx in range(len(gt_df)):
        gt_row = np.array(gt_flags.iloc[idx])
        pred_row = np.array(pred_flags.iloc[idx])
        weights_row = np.array(weight_flags.iloc[idx])
        if len(gt_row) != len(pred_row):
            raise ValueError("예측한 압력계 개수가 샘플 관망 구조와 다릅니다.")

        is_normal_sample = np.all(gt_row == 0)
        # 1) 정상 샘플에 대한 계산
        # -> 정상 샘플을 정상으로 잘 예측한 경우에는 점수 계산에 포함하지 않음
        # -> 정상 샘플을 비정상으로 잘못 예측한 경우에는 0점으로 반영 (패널티)
        if is_normal_sample:
            if np.sum(pred_row) > 0:  # False Positives
                valid_samples += 1  # Include in valid samples
                total_f1 += 0  # Penalize False Positives
            continue  # Skip further calculations for normal samples

        # 2) 비정상 샘플에 대한 계산
        # 정답과 예측이 동시에 1인 위치의 가중치 합
        matched_abnormal_weights = np.sum(weights_row * (gt_row == 1) * (pred_row == 1))
        # 예측값이 1인 위치의 가중치 합
        predicted_abnormal_weights = np.sum(weights_row * (pred_row == 1))
        # 정답 위치의 가중치 합
        total_abnormal_weights = np.sum(weights_row * (gt_row == 1))

        # False Positives: 정답이 0이고, 가중치도 0인데 예측이 1인 경우
        false_positives = np.sum((pred_row == 1) & (gt_row == 0) & (weights_row == 0))

        # Precision 계산: False Positives를 고려한 방식
        precision = (matched_abnormal_weights / (predicted_abnormal_weights + false_positives) 
                     if (predicted_abnormal_weights + false_positives) > 0 else None)
        recall = (matched_abnormal_weights / total_abnormal_weights
                  if total_abnormal_weights > 0 else None)

        if precision is not None and recall is not None and precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0

        total_f1 += f1_score
        valid_samples += 1

    average_f1 = total_f1 / valid_samples if valid_samples > 0 else 0
    return average_f1