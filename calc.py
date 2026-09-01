import pandas as pd
import requests
import xgboost as xgb
import numpy as np
import itertools
import os

def fetch_comprehensive_jolpica_history(start_year, end_year):
    """Fetches F1 history using pagination to bypass the 100-record limit."""
    all_races = []
    for season in range(start_year, end_year + 1):
        offset = 0
        while True:
            url = f"https://api.jolpi.ca/ergast/f1/{season}/results.json?limit=100&offset={offset}"
            try:
                response = requests.get(url, timeout=10)
                if response.status_code != 200: break
                data = response.json()
                races = data['MRData']['RaceTable']['Races']
                if not races: break
                
                for race in races:
                    round_no = int(race['round'])
                    circuit = race['Circuit']['circuitId']
                    
                    for result in race['Results']:
                        pos = result['position']
                        fin_pos = int(pos) if str(pos).isdigit() else 22
                        grid_raw = result.get('grid', 22)
                        grid_pos = int(grid_raw) if str(grid_raw).isdigit() and int(grid_raw) > 0 else 22
                        
                        all_races.append({
                            'season': int(season),
                            'round': round_no,
                            'circuit_id': circuit,
                            'driver_id': result['Driver']['driverId'],
                            'constructor_id': result['Constructor']['constructorId'],
                            'grid_position': grid_pos,
                            'finishing_position': fin_pos
                        })
                
                total = int(data["MRData"]["total"])
                offset += 100
                if offset >= total: break
            except Exception:
                break
                
    df = pd.DataFrame(all_races)
    if not df.empty:
        df = df.sort_values(by=['season', 'round']).reset_index(drop=True)
    return df

print("Fetching historical training data (2023–2026)...")
df_history = fetch_comprehensive_jolpica_history(2023, 2026)

# Base rolling calculations
df_history['driver_track_avg'] = df_history.groupby(['driver_id', 'circuit_id'])['finishing_position'].transform(
    lambda x: x.shift().expanding().mean()
).fillna(11.0)

df_history['driver_recent_form'] = df_history.groupby('driver_id')['finishing_position'].transform(
    lambda x: x.shift().rolling(window=3, min_periods=1).mean()
).fillna(11.0)

df_history['podium'] = df_history['finishing_position'].apply(lambda x: x <= 3)

# ==============================================================================
# OPTIMIZATION PARAMETER GRID
# ==============================================================================
param_grid = {
    'max_depth': [3, 4],
    'learning_rate': [0.03, 0.05],
    'n_estimators': [100, 150],
    'form_weight': [0.4, 0.5, 0.6] 
}

keys = param_grid.keys()
combinations = list(itertools.product(*param_grid.values()))

print(f"Testing {len(combinations)} hyperparameter configurations...")

best_score = float('inf')
best_params = None

# Broad validation pool across 2025 and 2026 rounds to capture more track varieties
validation_races = df_history[
    (df_history['season'] == 2025) | 
    ((df_history['season'] == 2026) & (df_history['round'] < 13))
][['season', 'round', 'circuit_id']].drop_duplicates()

print(f"Validation scope: Evaluating peak parameters across {len(validation_races)} historical rounds.")

for combo in combinations:
    params = dict(zip(keys, combo))
    
    max_d = params['max_depth']
    lr = params['learning_rate']
    n_est = params['n_estimators']
    f_weight = params['form_weight']
    t_weight = 1.0 - f_weight
    
    errors = []
    
    for _, row in validation_races.iterrows():
        t_season, t_round = row['season'], row['round']
        
        train_data = df_history[
            (df_history['season'] < t_season) | 
            ((df_history['season'] == t_season) & (df_history['round'] < t_round))
        ]
        
        if len(train_data) < 100: continue
        
        feature_cols = ['grid_position', 'driver_track_avg', 'driver_recent_form', 'constructor_id']
        X_train = pd.get_dummies(train_data[feature_cols], columns=['constructor_id'])
        y_train = train_data['podium'].astype(int)
        
        model = xgb.XGBClassifier(n_estimators=n_est, learning_rate=lr, max_depth=max_d, objective='binary:logistic', random_state=42)
        model.fit(X_train, y_train)
        
        test_data = df_history[(df_history['season'] == t_season) & (df_history['round'] == t_round)].copy()
        if test_data.empty: continue
        
        test_data['pace_score'] = (test_data['driver_recent_form'] * f_weight) + (test_data['driver_track_avg'] * t_weight)
        test_data = test_data.sort_values(by='pace_score').reset_index(drop=True)
        test_data['predicted_grid'] = test_data.index + 1
        test_data['grid_position'] = test_data['predicted_grid']
        
        X_test = pd.get_dummies(test_data[feature_cols], columns=['constructor_id'])
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        
        if X_test.empty: continue
        
        raw_probs = model.predict_proba(X_test)[:, 1]
        base_probs = raw_probs / np.sum(raw_probs) if np.sum(raw_probs) > 0 else raw_probs
        
        sim_ranks = np.zeros(len(test_data))
        for _ in range(200):
            noise = np.random.normal(0, 0.05, len(test_data))
            perf = np.clip(base_probs + noise, 0.001, None)
            order = np.argsort(perf)[::-1]
            for rank, idx in enumerate(order):
                sim_ranks[idx] += (rank + 1)
        
        test_data['model_pos'] = sim_ranks / 200
        test_data = test_data.sort_values(by='model_pos').reset_index(drop=True)
        test_data['predicted_rank'] = test_data.index + 1
        
        error = abs(test_data['predicted_rank'] - test_data['finishing_position']).mean()
        errors.append(error)
        
    if errors:
        mean_mae = np.mean(errors)
        if mean_mae < best_score:
            best_score = mean_mae
            best_params = params

print("\n==============================================")
print(f"🎯 OPTIMIZED PEAK PARAMETERS FOUND (2023–2026 Window):")
print(f"Best Configuration: {best_params}")
print(f"Aggregate Cross-Validation MAE: {best_score:.2f} positions")
print("==============================================")