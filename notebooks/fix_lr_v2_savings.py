import json, uuid

path = r'C:\Users\Miguel Fernández\Desktop\2º CURSO\SEGUNDO SEMESTRE\Aprendizaje Automático\Trabajo_Fraudes\notebooks\main.ipynb'

with open(path, encoding='utf-8') as f:
    nb = json.load(f)

# Verificar que la ultima celda es la de ahorro LR v2 incorrecta
last_src = ''.join(nb['cells'][-1].get('source', []))
print(f'Ultima celda empieza con: {last_src[:80]}')

savings_lr_v2_src = """\
# ============================================================
# AHORRO ECONOMICO — LogisticRegression + Selection V2
# ============================================================
df_sorted = df.sort_values('claim_datetime').reset_index(drop=True)
X_all = df_sorted.drop(columns=['fraud_flag'])
y_all = df_sorted['fraud_flag']

split_idx  = int(len(df_sorted) * 0.80)
X_train_ec = X_all.iloc[:split_idx]
X_test_ec  = X_all.iloc[split_idx:]
y_train_ec = y_all.iloc[:split_idx]
y_test_ec  = y_all.iloc[split_idx:]

print(f"Train: {len(X_train_ec)} muestras | Test: {len(X_test_ec)} muestras")
print(f"Fraudes en test: {y_test_ec.sum()} ({y_test_ec.mean()*100:.1f}%)")

best_params_lr_v2    = dict(study_lr.best_params)
smoothing_workshop   = best_params_lr_v2.pop('smoothing_workshop')
smoothing_make       = best_params_lr_v2.pop('smoothing_make')
smoothing_model      = best_params_lr_v2.pop('smoothing_model')
solver_v2            = best_params_lr_v2.pop('solver')
penalty_v2           = best_params_lr_v2.pop('penalty', 'l2')

best_model_lr_v2 = LogisticRegression(
    **best_params_lr_v2,
    solver=solver_v2, penalty=penalty_v2,
    random_state=42, n_jobs=-1
)
print(f"Modelo: LR V2 (solver={solver_v2}, penalty={penalty_v2})")
print(f"F-beta CV: {study_lr.best_value:.4f}")

best_estimator_lr_v2 = get_pipeline_with_selection_v2(best_model_lr_v2)
best_estimator_lr_v2.set_params(
    preprocess__feature_eng__trf_repair_workshop__smoothing=smoothing_workshop,
    preprocess__feature_eng__trf_make__smoothing=smoothing_make,
    preprocess__feature_eng__trf_model__smoothing=smoothing_model,
)
best_estimator_lr_v2.fit(X_train_ec, y_train_ec)
n_sel = best_estimator_lr_v2['selector'].get_support().sum()
print(f"Features seleccionadas: {n_sel}")

y_proba_test = best_estimator_lr_v2.predict_proba(X_test_ec)[:, 1]

C_PERITO  = 175
C_FRAUDE  = 2680
BETA_EC   = 3.78

n_fraud_test = y_test_ec.sum()
cost_base    = n_fraud_test * C_FRAUDE
print(f"Coste base (sin modelo): {cost_base:,.0f} EUR")

thresholds = np.arange(0.01, 0.99, 0.005)
metrics_by_threshold = []
for t in thresholds:
    y_pred_t = (y_proba_test >= t).astype(int)
    tn = ((y_test_ec == 0) & (y_pred_t == 0)).sum()
    fp = ((y_test_ec == 0) & (y_pred_t == 1)).sum()
    fn = ((y_test_ec == 1) & (y_pred_t == 0)).sum()
    tp = ((y_test_ec == 1) & (y_pred_t == 1)).sum()
    cost   = fp * C_PERITO + fn * C_FRAUDE + tp * C_PERITO
    saving = cost_base - cost
    metrics_by_threshold.append({
        'threshold': t, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'cost': cost, 'saving': saving,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall':    tp / (tp + fn) if (tp + fn) > 0 else 0,
        'fbeta':     fbeta_score(y_test_ec, y_pred_t, beta=BETA_EC)
    })

df_thresh = pd.DataFrame(metrics_by_threshold)
idx_best       = df_thresh['saving'].idxmax()
best_threshold = df_thresh.loc[idx_best, 'threshold']
best_saving    = df_thresh.loc[idx_best, 'saving']
best_cost      = df_thresh.loc[idx_best, 'cost']

print("")
print("=== UMBRAL OPTIMO ===")
print(f"Umbral:           {best_threshold:.3f}")
print(f"Coste con modelo: {best_cost:,.0f} EUR")
print(f"Ahorro neto:      {best_saving:,.0f} EUR  ({best_saving/cost_base*100:.1f}% del coste base)")
print("")
print("Detalle en test:")
print(confusion_matrix(y_test_ec, (y_proba_test >= best_threshold).astype(int)))
print("")
print("=== COMPARATIVA ===")
print(f"  Sin modelo:                         {cost_base:>10,.0f} EUR")
print(f"  XGBoost Optuna (base):              {420505:>10,.0f} EUR  (ahorro 16,335 EUR  3.7%)")
print(f"  LR Optuna (sin FE):                 {398695:>10,.0f} EUR  (ahorro 38,145 EUR  8.7%)")
print(f"  XGBoost + Selection (21 feat):      {416450:>10,.0f} EUR  (ahorro 20,475 EUR  4.7%)")
print(f"  XGBoost + Selection V2 (44 feat):   {424370:>10,.0f} EUR  (ahorro 12,470 EUR  2.9%)")
print(f"  LR + Selection V2 (este):           {best_cost:>10,.0f} EUR  (ahorro {best_saving:,.0f} EUR  {best_saving/cost_base*100:.1f}%)")\
"""

# Reemplazar la ultima celda
nb['cells'][-1]['source'] = savings_lr_v2_src
nb['cells'][-1]['outputs'] = []
nb['cells'][-1]['execution_count'] = None

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Celda de ahorro LR v2 corregida.')
new_src = ''.join(nb['cells'][-1].get('source', []))
print(f'Nueva ultima celda empieza con: {new_src[:80]}')
