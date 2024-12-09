import matplotlib.pyplot as plt
import os
import numpy as np

def plot_predict_actual(y_pred, y_test, model_name, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='GroundTruth', alpha=0.7)
    plt.plot(y_pred, label='Predict', alpha=0.7)
    plt.title(f'{model_name} - Predict')
    plt.xlabel('Sample')
    plt.ylabel('Target')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_predictpng'))
    plt.close()

def plot_feature_importance(model, feature_names, model_name, save_dir):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title(f'{model_name} Feature Importances')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_feature_importance.png'))
    plt.close()

def plot_feature_importance_linear(model, feature_names, model_name, save_dir):
    coef = model.coef_
    plt.figure(figsize=(10, 6))
    plt.title(f'{model_name} Coefficients')
    plt.bar(range(len(coef)), coef, align='center')
    plt.xticks(range(len(coef)), feature_names, rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_coefficients.png'))
    plt.close()