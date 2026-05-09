import json, uuid

path = r'C:\Users\Miguel Fernández\Desktop\2º CURSO\SEGUNDO SEMESTRE\Aprendizaje Automático\Trabajo_Fraudes\notebooks\main.ipynb'

with open(path, encoding='utf-8') as f:
    nb = json.load(f)

# Remove trailing empty cells
while nb['cells'] and not ''.join(nb['cells'][-1].get('source', [])).strip():
    nb['cells'].pop()

print(f'Celdas antes de añadir: {len(nb["cells"])}')

# ─── CELDA 1: definición de la pipeline v2 ───────────────────────────────────
pipeline_def_src = """\
# ============================================================
# PIPELINE V2 — FE activo + columnas extra de fraude_autoguard
# (las que ya venían en el dataset sin necesitar feature eng.)
# ============================================================

COLS_A_ELIMINAR_V2 = [
    # IDs y PII
    'claim_id', 'policy_id', 'customer_id', 'vehicle_id', 'license_plate',
    'full_name', 'email', 'phone', 'address', 'accident_description',
    # Fechas (ya procesadas por FE → driver_age, vehicle_age)
    'accident_datetime', 'date_of_birth', 'policy_start_date', 'policy_end_date',
    'last_address_change_date', 'manufacture_year',
    # city: cardinalidad demasiado alta para OHE
    'city',
]

COLS_NUM_V2 = [
    # Originales de get_pipeline_with_selection
    'number_of_supplements', 'claimed_amount_eur', 'deductible', 'annual_premium_eur',
    'number_of_cars', 'driver_age', 'vehicle_age', 'agent_claims_count',
    'workshop_fraud_rate', 'make_fraud_rate', 'model_fraud_rate',
    # Añadidas de fraude_autoguard (raw — ya en el dataset)
    'purchase_price_eur', 'odometer_km', 'driver_rating',
    'accident_latitude', 'accident_longitude', 'postal_code',
]

COLS_CAT_V2 = [
    # Originales de get_pipeline_with_selection
    'fault', 'police_report_filed', 'witness_present', 'past_number_of_claims',
    # Añadidas de fraude_autoguard (raw — ya en el dataset)
    'sex', 'marital_status', 'policy_type', 'base_policy',
    'vehicle_category', 'color', 'accident_area', 'province',
]


def get_features_preprocess_pipeline_v2():
    feature_eng_steps = Pipeline(steps=[
        ("trf_fechas",          FunctionTransformer(fechas_conductor_coche, validate=False)),
        ("trf_agent_id",        Agent_ID(apply_trf=True)),
        ("trf_repair_workshop", Repair_Workshop(apply_trf=True, smoothing=10)),
        ("trf_make",            MakeEncoder(apply_trf=True, smoothing=10)),
        ("trf_model",           ModelEncoder(apply_trf=True, smoothing=20)),
    ])

    # Igual que v1 pero SIN accident_latitude, accident_longitude, postal_code
    base_pandas_trf = ColumnTransformer(
        transformers=[
            ("drop",                "drop", COLS_A_ELIMINAR_V2),
            ("drop_claim_datetime", "drop", ["claim_datetime"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    transformer = ColumnTransformer(
        transformers=[
            ("num", num_transformer, COLS_NUM_V2),
            ("cat", cat_transformer, COLS_CAT_V2),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(steps=[
        ("feature_eng",        feature_eng_steps),
        ("trf",                base_pandas_trf),
        ("transformer",        transformer),
        ("variance_threshold", VarianceThreshold()),
    ])


def get_pipeline_with_selection_v2(model, threshold='median'):
    return Pipeline(steps=[
        ('preprocess', get_features_preprocess_pipeline_v2()),
        ('selector',   SelectFromModel(
                           XGBClassifier(n_estimators=100, max_depth=3,
                                         scale_pos_weight=scale_pos_weight,
                                         random_state=42, n_jobs=-1,
                                         eval_metric='logloss'),
                           threshold=threshold)),
        ('classifier', model),
    ])


# Validación rápida: cuántas features produce v2
_pipe_check = get_features_preprocess_pipeline_v2()
df_sorted_check = df.sort_values('claim_datetime').reset_index(drop=True)
_X_check = df_sorted_check.drop(columns=['fraud_flag'])
_y_check = df_sorted_check['fraud_flag']
_X_prep  = _pipe_check.fit(_X_check, _y_check).transform(_X_check)
print(f"Features pipeline v2: {_X_prep.shape[1]}")
del _pipe_check, _X_check, _y_check, _X_prep, df_sorted_check\
"""

# ─── CELDA 2: Optuna XGBoost con pipeline v2 ─────────────────────────────────
optuna_v2_src = """\
# ============================================================
# OPTUNA — XGBoost con get_pipeline_with_selection_v2
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


def objective_xgb_v2(trial):
    model = XGBClassifier(
        n_estimators      = trial.suggest_int('n_estimators', 100, 600, step=100),
        max_depth         = trial.suggest_int('max_depth', 3, 8),
        learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        subsample         = trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.6, 1.0),
        min_child_weight  = trial.suggest_int('min_child_weight', 1, 20),
        reg_alpha         = trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        reg_lambda        = trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        scale_pos_weight  = trial.suggest_float('scale_pos_weight', ratio * 0.6, ratio * 1.8),
        random_state      = 42,
        n_jobs            = -1,
        eval_metric       = 'logloss',
    )
    threshold = trial.suggest_categorical('sel_threshold', ['median', 'mean', '1.25*mean'])

    pipe = get_pipeline_with_selection_v2(model, threshold=threshold)

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


print(f"Optimizando XGBoost+Selection V2 (50 trials, F-beta={BETA})...")
t0 = time.time()
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_xgb_v2 = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
)
study_xgb_v2.optimize(objective_xgb_v2, n_trials=50, show_progress_bar=True)

print(f"Tiempo: {time.time()-t0:.1f}s")
print(f"Mejor F-beta Score {BETA}: {study_xgb_v2.best_value:.4f}")
print("Mejores hiperparametros:")
for k, v in study_xgb_v2.best_params.items():
    print(f"   {k}: {v}")\
"""

# ─── CELDA 3: Ahorro económico con pipeline v2 ────────────────────────────────
savings_v2_src = """\
# ============================================================
# AHORRO ECONOMICO — XGBoost + Selection V2
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

best_params_v2   = dict(study_xgb_v2.best_params)
sel_threshold_v2 = best_params_v2.pop('sel_threshold')

best_model_v2 = XGBClassifier(
    **best_params_v2,
    random_state=42, n_jobs=-1, eval_metric='logloss'
)
print(f"Modelo: XGBoost + Selection V2 (threshold={sel_threshold_v2})")
print(f"F-beta CV: {study_xgb_v2.best_value:.4f}")

best_estimator_v2 = get_pipeline_with_selection_v2(best_model_v2, threshold=sel_threshold_v2)
best_estimator_v2.fit(X_train_ec, y_train_ec)
n_sel = best_estimator_v2['selector'].get_support().sum()
print(f"Features seleccionadas: {n_sel}")

y_proba_test = best_estimator_v2.predict_proba(X_test_ec)[:, 1]

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
print(f"  Sin modelo:                      {cost_base:>10,.0f} EUR")
print(f"  XGBoost Optuna (base):           {420505:>10,.0f} EUR  (ahorro 16,335 EUR  3.7%)")
print(f"  LR Optuna (sin FE):              {398695:>10,.0f} EUR  (ahorro 38,145 EUR  8.7%)")
print(f"  XGBoost + Selection (21 feat):   {416450:>10,.0f} EUR  (ahorro 20,475 EUR  4.7%)")
print(f"  XGBoost + Selection V2 (este):   {best_cost:>10,.0f} EUR  (ahorro {best_saving:,.0f} EUR  {best_saving/cost_base*100:.1f}%)")\
"""


def make_cell(src, cid):
    return {'cell_type': 'code', 'execution_count': None, 'id': cid,
            'metadata': {}, 'outputs': [], 'source': src}


nb['cells'].append(make_cell(pipeline_def_src, str(uuid.uuid4())[:8]))
nb['cells'].append(make_cell(optuna_v2_src,    str(uuid.uuid4())[:8]))
nb['cells'].append(make_cell(savings_v2_src,   str(uuid.uuid4())[:8]))

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

with open(path, encoding='utf-8') as f:
    nb2 = json.load(f)
print(f'Total celdas guardadas: {len(nb2["cells"])}')
for i, cell in enumerate(nb2['cells'][-3:], start=len(nb2['cells'])-3):
    src = ''.join(cell.get('source', []))
    lines = [l for l in src.split('\n') if l.strip()]
    print(f'[{i}] {lines[0][:80]}')
