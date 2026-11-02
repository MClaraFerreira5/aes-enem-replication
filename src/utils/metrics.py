import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_squared_error

def compute_qwk(y_true, y_pred, min_rating=0, max_rating=1000):
    if len(y_true) == 0:
        return 0.0
    y_pred_discrete = np.round(y_pred).astype(int)
    y_true_discrete = np.round(y_true).astype(int)
    y_pred_discrete = np.clip(y_pred_discrete, min_rating, max_rating)
    
    try:
        return cohen_kappa_score(
            y_true_discrete, 
            y_pred_discrete, 
            weights="quadratic"
        )
    except Exception:
        return 0.0

def compute_rmse(y_true, y_pred):
    if len(y_true) == 0:
        return 0.0
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def compute_enem_discrepancy(y_true, y_pred, threshold=80.0):
    if len(y_true) == 0:
        return 0.0
    errors = np.abs(np.array(y_true) - np.array(y_pred))
    return float((errors > threshold).mean() * 100.0)

def evaluate_predictions(y_true, y_pred, target_name="Score"):
    if len(y_true) == 0:
        print(f"--- Resultados para {target_name} ---")
        print("Aviso: Conjunto de dados vazio (0 amostras).")
        return {"target": target_name, "qwk": 0.0, "rmse": 0.0, "discrepancy_gt80": 0.0}

    qwk = compute_qwk(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    discrepancy = compute_enem_discrepancy(y_true, y_pred)
    
    print(f"--- Resultados para {target_name} ---")
    print(f"QWK:                  {qwk:.4f}")
    print(f"RMSE:                 {rmse:.4f}")
    print(f"Discrepância (>80p):  {discrepancy:.2f}%")
    
    return {
        "target": target_name,
        "qwk": qwk,
        "rmse": rmse,
        "discrepancy_gt80": discrepancy
    }