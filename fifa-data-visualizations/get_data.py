import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import json

df = pd.read_csv('/Volumes/hard/fifa-data-visualizations/fifa_world_cup_2026_player_performance.csv')

# Drop unused
drop_cols = ['player_id', 'match_id', 'player_name', 'match_date', 'stadium', 'city', 'opponent_team', 'tournament_stage', 'club_name', 'nationality', 'team', 'position', 'preferred_foot']
X = df.drop(columns=['match_result'] + drop_cols, errors='ignore')
y = df['match_result']

# Basic cleaning
X = X.select_dtypes(include=[np.number])
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_encoded, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)

acc = rf.score(X_test, y_test)
importances = rf.feature_importances_

feature_names = X.columns
imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False).head(5)

# Stage distribution
stage_dist = df['tournament_stage'].value_counts().to_dict()

# Positional distribution
pos_dist = df['position'].value_counts().to_dict()

out = {
    'total_records': len(df),
    'accuracy': round(acc * 100, 1),
    'top_features': imp_df.to_dict('records'),
    'stage_dist': stage_dist,
    'pos_dist': pos_dist
}

with open('/Volumes/hard/fifa-data-visualizations/dashboard_data.json', 'w') as f:
    json.dump(out, f)
print("Data dumped")
