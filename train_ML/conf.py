param_grids = {
    'GradientBoostingRegressor': {
        'n_estimators': [100, 500, 1000, 1500, 2000, 2500, 3000],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    },
    'RandomForestRegressor': {
        'n_estimators': [100, 500, 1000, 1500, 2000, 2500, 3000],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'LinearRegression': {},  # 기본 파라미터
    'DecisionTreeRegressor': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'XGBRegressor': {
        'n_estimators': [100, 500, 1000, 1500, 2000, 2500, 3000],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'tree_method': ['gpu_hist'],  # GPU 사용
        'gpu_id': [0]  # GPU 장치 ID
    },
    'LGBMRegressor': {
        'n_estimators': [100, 500, 1000, 1500, 2000, 2500, 3000],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [-1, 10, 20, 30],
        'num_leaves': [31, 50, 70],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'device': ['gpu'],
        'gpu_device_id': [0]
    },
    'KNeighborsRegressor': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    'SVR': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    },
    'ExtraTreesRegressor': {
        'n_estimators': [100, 500, 1000, 1500, 2000, 2500, 3000],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'AdaBoostRegressor': {
        'n_estimators': [50, 100, 200, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0]
    },
    'XGBRFRegressor': {
        'n_estimators': [100, 500, 1000, 1500, 2000, 2500, 3000],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'tree_method': ['gpu_hist'],  # GPU 사용
        'gpu_id': [0]  # GPU 장치 ID
    }
}
