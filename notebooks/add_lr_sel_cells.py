import json, uuid

path = r'C:\Users\Miguel Fernández\Desktop\2º CURSO\SEGUNDO SEMESTRE\Aprendizaje Automático\Trabajo_Fraudes\notebooks\main.ipynb'

with open(path, encoding='utf-8') as f:
    nb = json.load(f)

# Remove trailing empty cells
while nb['cells'] and not ''.join(nb['cells'][-1].get('source', [])).strip():
    removed = nb['cells'].pop()
    print(f'Removed empty cell: {removed.get("id")}')

print(f'Cells before adding: {len(nb["cells"])}')

optuna_src = """\
# ============================================================
# OPTUNA - LogisticRegression con get_pipeline_with_selection (FE activo)
# ============================================================
df_sorted = df.sort_values('claim_datetime').reset_index(drop=True)
X      = df_sorted.drop(columns=['fraud_flag'])
y      = df_sorted['fraud_flag']
groups = df_sorted['customer_id']
ratio  = len(y[y == 0]) / max(len(y[y == 1]), 1)
scale_pos_weight = int(ratio)
print(f"Ratio clases (neg/pos): {ratio:.2f}")
BETA       = 3.78
MIN_RECALL = 0.20
PENALTY    = 0.80
N_SPLITS   = 5

def objective_lr_sel(trial):
    solver  = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga'])
    penalty = 'l2' if solver == 'lbfgs' else trial.suggest_categorical('penalty', ['l1', 'l2'])

    model = LogisticRegression(
        C            = trial.suggest_float('C', 1e-4, 10.0, log=True),
        penalty      = penalty,
        solver       = solver,
        max_iter     = trial.suggest_int('max_iter', 100, 1000, step=100),
        class_weight = trial.suggest_categorical('class_weight', [None, 'balanced']),
        tol          = trial.suggest_float('tol', 1e-5, 1e-3, log=True),
        random_state = 42,
        n_jobs       = -1
    )
    threshold = trial.suggest_categorical('sel_threshold', ['median', 'mean', '1.25*mean'])

    pipe = get_pipeline_with_selection(model, threshold=threshold)

    tscv = GroupTimeSeriesSplit(n_splits=N_SPLITS)
    fold_scores = []

    for fold_i, (train_idx, valid_idx) in enumerate(tscv.split(X, y, groups), start=1):
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]

        pipe.fit(X_tr, y_tr)
        y_proba_va = pipe.predict_proba(X_va)[:, 1]
        y_pred_va  = (y_proba_va >= 0.5).astype(int)

        fb  = fbeta_score(y_va, y_pred_va, beta=BETA, zero_division=0)
        rec = ((y_va == 1) & (y_pred_va == 1)).sum() / max((y_va == 1).sum(), 1)
        fold_scores.append(fb - max(0.0, MIN_RECALL - rec) * PENALTY)

        trial.report(float(np.mean(fold_scores)), step=fold_i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_scores))


print(f"Optimizando LR+Selection con Optuna (50 trials, F-beta={BETA})...")
t0 = time.time()
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_lr_sel = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
)
study_lr_sel.optimize(objective_lr_sel, n_trials=50, show_progress_bar=True)

print(f"Tiempo: {time.time()-t0:.1f}s")
print(f"Mejor F-beta Score {BETA}: {study_lr_sel.best_value:.4f}")
print("Mejores hiperparametros:")
for k, v in study_lr_sel.best_params.items():
    print(f"   {k}: {v}")"""

savings_src = """\
# ============================================================
# AHORRO ECONOMICO - LogisticRegression + Selection (FE activo)
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

best_params_lr_sel = dict(study_lr_sel.best_params)
sel_threshold      = best_params_lr_sel.pop('sel_threshold')
solver             = best_params_lr_sel.pop('solver')
penalty            = best_params_lr_sel.pop('penalty', 'l2')

best_model_lr_sel = LogisticRegression(
    **best_params_lr_sel,
    solver=solver, penalty=penalty,
    random_state=42, n_jobs=-1
)
print(f"Modelo: LR + Selection (solver={solver}, penalty={penalty}, threshold={sel_threshold})")
print(f"F-beta CV: {study_lr_sel.best_value:.4f}")

best_estimator_lr_sel = get_pipeline_with_selection(best_model_lr_sel, threshold=sel_threshold)
best_estimator_lr_sel.fit(X_train_ec, y_train_ec)
n_sel = best_estimator_lr_sel['selector'].get_support().sum()
print(f"Features seleccionadas: {n_sel}")

y_proba_test = best_estimator_lr_sel.predict_proba(X_test_ec)[:, 1]

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
print(f"  Sin modelo:                  {cost_base:>10,.0f} EUR")
print(f"  XGBoost Optuna (base):       {420505:>10,.0f} EUR  (ahorro 16,335 EUR  3.7%)")
print(f"  LR Optuna (sin FE):          {398695:>10,.0f} EUR  (ahorro 38,145 EUR  8.7%)")
print(f"  LR + Selection (este):       {best_cost:>10,.0f} EUR  (ahorro {best_saving:,.0f} EUR  {best_saving/cost_base*100:.1f}%)")"""


def make_cell(src, cid):
    return {'cell_type': 'code', 'execution_count': None, 'id': cid,
            'metadata': {}, 'outputs': [], 'source': src}

nb['cells'].append(make_cell(optuna_src,  str(uuid.uuid4())[:8]))
nb['cells'].append(make_cell(savings_src, str(uuid.uuid4())[:8]))

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

# Verify
with open(path, encoding='utf-8') as f:
    nb2 = json.load(f)
print(f'Total celdas guardadas: {len(nb2["cells"])}')
for i, cell in enumerate(nb2['cells'][-4:], start=len(nb2['cells'])-4):
    src = ''.join(cell.get('source', []))
    lines = [l for l in src.split('\n') if l.strip()]
    print(f'[{i}] {lines[1][:70] if len(lines) > 1 else lines[0][:70]}')
