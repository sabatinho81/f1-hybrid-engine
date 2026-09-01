import pandas as pd
import requests
import xgboost as xgb
import numpy as np
import fastf1
import os

# --- Enable FastF1 Cache ---
os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")

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

def run_monte_carlo(model, feature_matrix, n_sims=5000):
    """Simulates thousands of races to return expected finishing positions."""
    raw_probs = model.predict_proba(feature_matrix)[:, 1]
    base_probs = raw_probs / np.sum(raw_probs)
    n_drivers = len(feature_matrix)
    
    finishing_positions_sum = np.zeros(n_drivers)

    for _ in range(n_sims):
        noise = np.random.normal(0, 0.05, n_drivers)
        performance = np.clip(base_probs + noise, 0.001, None)
        race_order = np.argsort(performance)[::-1]
        
        for idx, driver_idx in enumerate(race_order):
            finishing_positions_sum[driver_idx] += (idx + 1)

    return finishing_positions_sum / n_sims

print("Fetching historical data from Jolpica...")
df_history = fetch_comprehensive_jolpica_history(2021, 2026)

# Calculate rolling averages
df_history['driver_track_avg'] = df_history.groupby(['driver_id', 'circuit_id'])['finishing_position'].transform(
    lambda x: x.shift().expanding().mean()
).fillna(11.0)

df_history['driver_recent_form'] = df_history.groupby('driver_id')['finishing_position'].transform(
    lambda x: x.shift().rolling(window=3, min_periods=1).mean()
).fillna(11.0)

df_history['podium'] = df_history['finishing_position'].apply(lambda x: 1 if x <= 3 else 0)

# --- BACKTEST TARGET: 2026 DUTCH GP (Round 12) ---
target_circuit = 'zandvoort'
target_round = 12
target_year = 2026

# Train strictly on races BEFORE the 2026 Dutch GP
train_data = df_history[
    (df_history['season'] < target_year) | 
    ((df_history['season'] == target_year) & (df_history['round'] < target_round))
]

feature_cols = ['grid_position', 'driver_track_avg', 'driver_recent_form', 'constructor_id']
X_train = pd.get_dummies(train_data[feature_cols], columns=['constructor_id'])
y_train = train_data['podium']

model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.03, max_depth=4, objective='binary:logistic')
model.fit(X_train, y_train)

# 2026 Active Grid Roster
active_2026_grid = [
    {'driver_id': 'max_verstappen', 'constructor_id': 'red_bull'},
    {'driver_id': 'hadjar', 'constructor_id': 'red_bull'},
    {'driver_id': 'norris', 'constructor_id': 'mclaren'},
    {'driver_id': 'piastri', 'constructor_id': 'mclaren'},
    {'driver_id': 'leclerc', 'constructor_id': 'ferrari'},
    {'driver_id': 'hamilton', 'constructor_id': 'ferrari'},
    {'driver_id': 'russell', 'constructor_id': 'mercedes'},
    {'driver_id': 'antonelli', 'constructor_id': 'mercedes'},
    {'driver_id': 'sainz', 'constructor_id': 'williams'},
    {'driver_id': 'albon', 'constructor_id': 'williams'},
    {'driver_id': 'alonso', 'constructor_id': 'aston_martin'},
    {'driver_id': 'stroll', 'constructor_id': 'aston_martin'},
    {'driver_id': 'gasly', 'constructor_id': 'alpine'},
    {'driver_id': 'colapinto', 'constructor_id': 'alpine'},
    {'driver_id': 'ocon', 'constructor_id': 'haas'},
    {'driver_id': 'bearman', 'constructor_id': 'haas'},
    {'driver_id': 'hulkenberg', 'constructor_id': 'audi'},
    {'driver_id': 'bortoleto', 'constructor_id': 'audi'},
    {'driver_id': 'lawson', 'constructor_id': 'racing_bulls'},
    {'driver_id': 'lindblad', 'constructor_id': 'racing_bulls'},
    {'driver_id': 'perez', 'constructor_id': 'cadillac'},
    {'driver_id': 'bottas', 'constructor_id': 'cadillac'}
]

target_race_data = df_history[(df_history['season'] == target_year) & (df_history['round'] == target_round)]

prediction_rows = []
for entry in active_2026_grid:
    d_id = entry['driver_id']
    driver_stats = target_race_data[target_race_data['driver_id'] == d_id]
    
    t_avg = driver_stats.iloc[0]['driver_track_avg'] if not driver_stats.empty else 11.0
    r_form = driver_stats.iloc[0]['driver_recent_form'] if not driver_stats.empty else 11.0
    pace_score = (r_form * 0.6) + (t_avg * 0.4)
    
    prediction_rows.append({
        'driver_id': d_id,
        'constructor_id': entry['constructor_id'],
        'driver_track_avg': t_avg,
        'driver_recent_form': r_form,
        'pace_score': pace_score
    })

base_df = pd.DataFrame(prediction_rows)

# Fetch Actual Race Results for Comparison
race_session = fastf1.get_session(target_year, target_round, 'R')
race_session.load(telemetry=False, weather=False, messages=False)

actual_results = {}
for index, row in race_session.results.iterrows():
    driver_ref = str(row['LastName']).lower()
    actual_pos = row['Position']
    actual_results[driver_ref] = int(actual_pos) if pd.notna(actual_pos) else 22

def get_actual_finish(d_id):
    for ref, pos in actual_results.items():
        if ref in d_id:
            return pos
    return 22

base_df['Actual_Finish'] = base_df['driver_id'].apply(get_actual_finish)

# ==============================================================================
# PIPELINE A: SIMULATION USING PREDICTED QUALIFYING GRID
# ==============================================================================
pred_quali_df = base_df.sort_values(by='pace_score').reset_index(drop=True)
pred_quali_df['grid_position'] = pred_quali_df.index + 1

X_pred_quali = pd.get_dummies(pred_quali_df[feature_cols], columns=['constructor_id'])
X_pred_quali = X_pred_quali.reindex(columns=X_train.columns, fill_value=0)

pred_quali_df['Model_Exp_Finish'] = run_monte_carlo(model, X_pred_quali, n_sims=5000)
pred_quali_df = pred_quali_df.sort_values(by='Model_Exp_Finish').reset_index(drop=True)
pred_quali_df['Predicted_Rank'] = pred_quali_df.index + 1

errors_a = abs(pred_quali_df['Predicted_Rank'] - pred_quali_df['Actual_Finish'])
mae_a = np.mean(errors_a)

# ==============================================================================
# PIPELINE B: SIMULATION USING REAL FASTF1 QUALIFYING GRID
# ==============================================================================
quali_session = fastf1.get_session(target_year, target_round, 'Q')
quali_session.load(telemetry=False, weather=False, messages=False)

real_grid_map = {}
for index, row in quali_session.results.iterrows():
    driver_ref = str(row['LastName']).lower()
    real_grid_map[driver_ref] = row['Position']

real_quali_df = base_df.copy()
def get_real_grid(d_id):
    for ref, pos in real_grid_map.items():
        if ref in d_id:
            return pos
    return 22

real_quali_df['grid_position'] = real_quali_df['driver_id'].apply(get_real_grid)

X_real_quali = pd.get_dummies(real_quali_df[feature_cols], columns=['constructor_id'])
X_real_quali = X_real_quali.reindex(columns=X_train.columns, fill_value=0)

real_quali_df['Model_Exp_Finish'] = run_monte_carlo(model, X_real_quali, n_sims=5000)
real_quali_df = real_quali_df.sort_values(by='Model_Exp_Finish').reset_index(drop=True)
real_quali_df['Predicted_Rank'] = real_quali_df.index + 1

errors_b = abs(real_quali_df['Predicted_Rank'] - real_quali_df['Actual_Finish'])
mae_b = np.mean(errors_b)

# ==============================================================================
# SIDE-BY-SIDE COMPARISON OUTPUT
# ==============================================================================
print(f"\n--- SIDE-BY-SIDE BACKTEST COMPARISON (2026 Dutch GP) ---")

comparison_matrix = pd.DataFrame({
    'Driver': real_quali_df['driver_id'],
    'Actual Finish': real_quali_df['Actual_Finish'].apply(lambda x: x if x != 22 else 'DNF'),
    'Pred-Grid Sim Rank': pred_quali_df.set_index('driver_id').loc[real_quali_df['driver_id']]['Predicted_Rank'].values,
    'Real-Grid Sim Rank': real_quali_df['Predicted_Rank'].values
})

print(comparison_matrix.to_string(index=False))

print("\n--------------------------------------------------------")
print(f"🎯 MAE Using Model's Predicted Qualifying Grid: {mae_a:.2f} positions")
print(f"🎯 MAE Using Real FastF1 Qualifying Grid:     {mae_b:.2f} positions")
print("--------------------------------------------------------")