import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_squared_error, accuracy_score

def compute_qwk(y_true, y_pred, min_rating=0, max_rating=200):
    """
    Calcula o Quadratic Weighted Kappa (QWK) entre as notas reais e preditas.
    Garante que os valores estejam discretizados na escala (ex: passos de 20 para ENEM ou inteiros).
    """
    # Arredondar predições contínuas para inteiros / classes
    y_pred_discrete = np.round(y_pred).astype(int)
    y_true_discrete = np.round(y_true).astype(int)
    
    # Clipar para os limites válidos da escala
    y_pred_discrete = np.clip(y_pred_discrete, min_rating, max_rating)
    
    return cohen_kappa_score(
        y_true_discrete, 
        y_pred_discrete, 
        weights="quadratic", 
        labels=list(range(min_rating, max_rating + 1))
    )

def compute_rmse(y_true, y_pred):
    """
    Calcula a Raiz do Erro Quadrático Médio (RMSE).
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def compute_enem_discrepancy(y_true, y_pred, threshold=80.0):
    """
    Calcula a porcentagem de casos onde o erro absoluto é superior a 'threshold' pontos
    (No ENEM, divergências acima de 80 pontos na nota final levam a redação para um 3º corretor).
    """
    errors = np.abs(np.array(y_true) - np.array(y_pred))
    discrepancies = (errors > threshold).mean() * 100.0
    return discrepancies

def evaluate_predictions(y_true, y_pred, target_name="Score"):
    """
    Consolida todas as métricas principais da Fase 1 em um único dicionário.
    """
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

if __name__ == "__main__":
    # Teste rápido de validação do script
    y_real = [120, 160, 80, 200, 140]
    y_chutado = [120, 140, 100, 180, 140]
    
    metrics = evaluate_predictions(y_real, y_chutado, "Teste Interno")