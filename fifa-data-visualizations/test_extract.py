import pandas as pd
import json

df = pd.read_csv('/Volumes/hard/fifa-data-visualizations/fifa_world_cup_2026_player_performance.csv')

# 1. Top 10 players by rating
top_players = df.groupby('player_name')['tournament_rating'].mean().sort_values(ascending=False).head(10).to_dict()

# 2. Goals vs Assists (Sample 200 players)
ga_df = df[['goals', 'assists', 'player_name', 'position']].dropna().sample(min(200, len(df)), random_state=42)
ga_data = ga_df.to_dict('records')

# 3. Match Results count
match_results = df['match_result'].value_counts().to_dict()

# 4. Avg Stamina by Position
stamina = df.groupby('position')['stamina_score'].mean().to_dict()

# 5. Top speed distribution (histogram counts)
speed_hist = pd.cut(df['top_speed_kmh'], bins=10).value_counts().sort_index()
speed_bins = [str(i) for i in speed_hist.index]
speed_counts = speed_hist.values.tolist()

out = {
    'top_players': top_players,
    'ga_data': ga_data,
    'match_results': match_results,
    'stamina': stamina,
    'speed_bins': speed_bins,
    'speed_counts': speed_counts
}
print(list(out.keys()))
