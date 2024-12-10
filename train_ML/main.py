import pandas as pd
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm
import os
from typing import List, Dict, Union
from data_processing import *
from plotting import *
from conf import param_grids

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor, XGBRFRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

def get_labels(df: pd.DataFrame, window_size: int, stride: int, valid_starts: list) -> np.ndarray:
    """유효한 윈도우의 레이블을 추출"""
    accident_labels = df['anomaly'].values
    labels = np.array([accident_labels[idx + window_size] for idx in valid_starts])
    return labels

def train(save_dir, x_train, y_train, x_test, y_test):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_file = open(os.path.join(save_dir, "model_training_log.txt"), "w")

    # 사용하려는 모든 모델 리스트 정의
    models = [
        GradientBoostingRegressor(),
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
        XGBRegressor(),
        LGBMRegressor(verbose=-1),
        KNeighborsRegressor(),
        SVR(),
        ExtraTreesRegressor(),
        AdaBoostRegressor(),
        XGBRFRegressor()
    ]

    results = {
        'Model_Name': [],
        'R2_score': [],
        'RMSE': [],
        'NMAE': [],
        'Best_Params': [],
        'Predict': []
    }

    for model in models:
        model_name_current = model.__class__.__name__
        log_file.write(f"====================\n")
        log_file.write(f"Processing {model_name_current}\n")
        log_file.write(f"====================\n")
        
        param_grid = param_grids.get(model_name_current, {})
        
        # GridSearchCV 설정
        if param_grid:
            search = GridSearchCV(
                estimator=model, 
                param_grid=param_grid, 
                cv=3, 
                n_jobs=-1, 
                verbose=2,
                scoring='r2'  # R² 점수를 기준으로 평가
            )
        else:
            # 파라미터 그리드가 없는 모델은 그대로 사용
            search = model

        # 하이퍼파라미터 탐색
        if param_grid:
            log_file.write(f"Starting GridSearchCV for {model_name_current}\n")
            search.fit(x_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            log_file.write(f"Best parameters for {model_name_current}: {best_params}\n")
        else:
            log_file.write(f"No hyperparameter tuning for {model_name_current}. Training directly.\n")
            search.fit(x_train, y_train)
            best_model = search
            best_params = {}

        # 예측 수행
        y_pred = best_model.predict(x_test)

        # 성능 지표 계산
        r2 = r2_score(y_test, y_pred) * 100
        rmse_value = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        nmae_value = mae / y_test.mean()

        # 결과 저장
        results['Model_Name'].append(model_name_current)
        results['R2_score'].append(r2)
        results['RMSE'].append(rmse_value)
        results['NMAE'].append(nmae_value)
        results['Best_Params'].append(best_params)
        results['Predict'].append(y_pred)

        # 모델 저장
        joblib.dump(best_model, os.path.join(save_dir, f'{model_name_current}.pkl'))

        # 예측 결과 시각화
        plot_predict_actual(y_pred, y_test, model_name_current, save_dir)

        # 로그 기록
        log_file.write(f"R2 Score: {r2:.2f}\n")
        log_file.write(f"RMSE: {rmse_value:.2f}\n")
        log_file.write(f"NMAE: {nmae_value:.4f}\n")
        log_file.write(f"Best Hyperparameters: {best_params}\n\n")

        # 특성 중요도 시각화 (지원하는 모델에 한함)
        if hasattr(best_model, 'feature_importances_'):
            plot_feature_importance(best_model, x_train.columns, model_name_current, save_dir)
        elif hasattr(best_model, 'coef_'):
            plot_feature_importance_linear(best_model, x_train.columns, model_name_current, save_dir)

    # 결과 DataFrame 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_dir, "model_scores.csv"), index=False)

    log_file.close()

    # R2 점수가 가장 높은 모델 선택
    best_model_info = results_df.loc[results_df['R2_score'].idxmax()]
    print(f"Best Model: {best_model_info['Model_Name']} with R2 Score: {best_model_info['R2_score']:.2f}%")
    
    # 최적 모델 반환 (필요시)
    return best_model_info



#########################################
if __name__ == "__main__":    
    df_A = pd.read_csv("../data/train/TRAIN_A.csv")
    df_B = pd.read_csv("../data/train/TRAIN_B.csv")

    merged_df = merge_datasets([df_A, df_B])

    # 학습 데이터 전처리
    train_windows = prepare_training_data(
        df=merged_df,
        window_size=CFG.WINDOW_GIVEN,
        stride=CFG.STRIDE
    ) # Shape: (num_train_windows, window_size, num_features)

    # 윈도우 인덱스 생성
    num_total = train_windows.shape[0]
    indices = np.arange(num_total)

    # 학습용과 테스트용 인덱스 분할
    train_indices, test_indices = train_test_split(
        indices,
        test_size=CFG.TEST_SIZE,
        random_state=CFG.RANDOM_STATE,
        shuffle=True
    )

    # 유효한 시작 인덱스 계산
    potential_starts = np.arange(0, len(merged_df) - CFG.WINDOW_GIVEN, CFG.STRIDE)
    accident_labels = merged_df['anomaly'].values
    valid_starts = [
        idx for idx in potential_starts 
        if (idx + CFG.WINDOW_GIVEN < len(merged_df)) and (accident_labels[idx + CFG.WINDOW_GIVEN] == 0)
    ]

    labels = get_labels(merged_df, CFG.WINDOW_GIVEN, CFG.STRIDE, valid_starts)

    x_train, x_test, y_train, y_test = train_test_split(
        train_windows, labels, test_size=CFG.TEST_SIZE, random_state=CFG.RANDOM_STATE, shuffle=True
    )

    x_train = pd.DataFrame(x_train.reshape(x_train.shape[0], -1))
    x_test = pd.DataFrame(x_test.reshape(x_test.shape[0], -1))

    x_train = x_train.interpolate(method='linear', axis=0).fillna(method='bfill').fillna(method='ffill')
    x_test = x_test.interpolate(method='linear', axis=0).fillna(method='bfill').fillna(method='ffill')

    save_dir = 'train_models'
    best_model_info = train(save_dir, x_train, y_train, x_test, y_test)