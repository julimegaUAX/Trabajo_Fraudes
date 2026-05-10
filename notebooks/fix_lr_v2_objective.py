import json

path = r'C:\Users\Miguel Fernández\Desktop\2º CURSO\SEGUNDO SEMESTRE\Aprendizaje Automático\Trabajo_Fraudes\notebooks\main.ipynb'

with open(path, encoding='utf-8') as f:
    nb = json.load(f)

# Celda 216 (índice -2) es el objetivo LR v2
target_cell = nb['cells'][-2]
src = ''.join(target_cell.get('source', []))
print('Primeras líneas de la celda a corregir:')
print(src[:200])
print('...')

correct_src = """\
# ============================================================
# OPTUNA — LogisticRegression con get_pipeline_with_selection_v2
# ============================================================
df_sorted = df.sort_values('claim_datetime').reset_index(drop=True)
X      = df_sorted.drop(columns=['fraud_flag'])
y      = df_sorted['fraud_flag']
groups = df_sorted['customer_id']
ratio  = len(y[y == 0]) / max(len(y[y == 1]), 1)
scale_pos_weight = int(ratio)
BETA       = 3.78
MIN_RECALL = 0.20
PENALTY    = 0.80
N_SPLITS   = 5


def objective_lr_v2(trial):
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
        n_jobs       = -1,
    )

    smoothing_workshop = trial.suggest_int('smoothing_workshop', 5, 50)
    smoothing_make     = trial.suggest_int('smoothing_make',     5, 50)
    smoothing_model    = trial.suggest_int('smoothing_model',    5, 50)

    pipe = get_pipeline_with_selection_v2(model)
    pipe.set_params(
        preprocess__feature_eng__trf_repair_workshop__smoothing=smoothing_workshop,
        preprocess__feature_eng__trf_make__smoothing=smoothing_make,
        preprocess__feature_eng__trf_model__smoothing=smoothing_model,
    )

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


print(f"Optimizando LR V2 con Optuna (50 trials, F-beta={BETA})...")
t0 = time.time()
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_lr_v2 = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
)
study_lr_v2.optimize(objective_lr_v2, n_trials=50, show_progress_bar=True)

print(f"Tiempo: {time.time()-t0:.1f}s")
print(f"Mejor F-beta Score {BETA}: {study_lr_v2.best_value:.4f}")
print("Mejores hiperparametros:")
for k, v in study_lr_v2.best_params.items():
    print(f"   {k}: {v}")\
"""

target_cell['source'] = correct_src
target_cell['outputs'] = []
target_cell['execution_count'] = None

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('\nCelda objetivo LR v2 corregida.')
print('Cambios aplicados:')
print('  - Añadido bloque de inicializacion (df_sorted, X, y, groups, ratio...)')
print('  - Renombrado objective_lr -> objective_lr_v2')
print('  - Corregido tscv.split(X, groups=groups) -> tscv.split(X, y, groups)')
print('  - Renombrado study_lr -> study_lr_v2 (evita colision con el estudio anterior)')
