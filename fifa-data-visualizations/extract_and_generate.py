import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

print("Loading data...")
df = pd.read_csv('/Volumes/hard/fifa-data-visualizations/fifa_world_cup_2026_player_performance.csv')

print("Training model...")
drop_cols = ['player_id', 'match_id', 'player_name', 'match_date', 'stadium', 'city', 'opponent_team', 'tournament_stage', 'club_name', 'nationality', 'team', 'position', 'preferred_foot']
X = df.drop(columns=['match_result'] + drop_cols, errors='ignore').select_dtypes(include=[np.number])
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
y_encoded = LabelEncoder().fit_transform(df['match_result'])

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_encoded, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)
acc = rf.score(X_test, y_test)
imp_df = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False).head(5)

print("Aggregating stats...")
# 1. Top Players
top_players_df = df.groupby('player_name')[['tournament_rating', 'offensive_contribution', 'defensive_contribution']].mean().sort_values('tournament_rating', ascending=False).head(8)
top_players_names = top_players_df.index.tolist()
top_off = top_players_df['offensive_contribution'].tolist()
top_def = top_players_df['defensive_contribution'].tolist()

# 2. Bubble Chart: Goals vs Assists vs Rating
ga_df = df[(df['goals'] > 0) | (df['assists'] > 0)][['goals', 'assists', 'player_name', 'position', 'tournament_rating']].dropna().sample(min(150, len(df)), random_state=42)
min_r, max_r = ga_df['tournament_rating'].min(), ga_df['tournament_rating'].max()
ga_df['r'] = ((ga_df['tournament_rating'] - min_r) / (max_r - min_r) * 12 + 4).round(1)
ga_data = ga_df.to_dict('records')

# 3. Match Outcomes
match_results = df['match_result'].value_counts().to_dict()

# 4. Positional Radar Chart
pos_radar = df.groupby('position')[['offensive_contribution', 'defensive_contribution', 'stamina_score', 'top_speed_kmh', 'tournament_rating']].mean()
for col in pos_radar.columns:
    pos_radar[col] = (pos_radar[col] / pos_radar[col].max() * 100).round(1)
radar_labels = ['Offense', 'Defense', 'Stamina', 'Speed', 'Overall Rating']
radar_datasets = []
for pos in pos_radar.index:
    if pos in ['Forward', 'Midfielder', 'Defender']:
        radar_datasets.append({
            'label': pos,
            'data': pos_radar.loc[pos].tolist()
        })

# 5. Feature Importances
rf_features = imp_df['feature'].str.replace('_', ' ').str.title().tolist()
rf_importances = (imp_df['importance'] * 100).round(1).tolist()

# 6. Age vs Rating
age_trend = df.groupby('age')['tournament_rating'].mean().round(2).sort_index()
age_labels = age_trend.index.astype(str).tolist()
age_ratings = age_trend.values.tolist()

data = {
    'total_records': len(df),
    'accuracy': round(acc * 100, 1),
    'top_names': top_players_names,
    'top_off': top_off,
    'top_def': top_def,
    'ga_data': ga_data,
    'match_results': match_results,
    'radar_labels': radar_labels,
    'radar_datasets': radar_datasets,
    'rf_features': rf_features,
    'rf_importances': rf_importances,
    'age_labels': age_labels,
    'age_ratings': age_ratings
}

print("Extracting images from model.ipynb...")
with open('/Volumes/hard/fifa-data-visualizations/model.ipynb', 'r') as f:
    nb = json.load(f)

# Complete Mapping of images in model.ipynb by cell index
image_mapping = {
    14: [{"title": "Missing Values Heatmap", "category": "demographics"}],
    15: [{"title": "Player Age Distribution", "category": "demographics"}],
    16: [{"title": "Positional Distribution", "category": "demographics"}],
    17: [{"title": "Preferred Foot Breakdown", "category": "demographics"}],
    18: [{"title": "Top 10 Player Nationalities", "category": "demographics"}],
    19: [{"title": "Top 10 Represented Clubs", "category": "demographics"}],
    20: [{"title": "Player Market Value Distribution", "category": "finances"}],
    21: [{"title": "Most Valuable Players", "category": "finances"}],
    22: [{"title": "Tournament Goals Distribution", "category": "finances"}],
    23: [{"title": "Top Tournament Scorers", "category": "finances"}],
    24: [{"title": "Top Playmakers (Assists)", "category": "finances"}],
    25: [{"title": "Player Rating Distribution", "category": "physicals"}],
    26: [{"title": "Performance Score Boxplot", "category": "physicals"}],
    27: [{"title": "Match Results Distribution", "category": "model"}],
    28: [{"title": "Tournament Stage Distribution", "category": "model"}],
    29: [{"title": "Goals vs Assists Scatterplot", "category": "finances"}],
    30: [{"title": "Minutes Played vs Player Rating", "category": "finances"}],
    31: [{"title": "Market Value vs Rating", "category": "finances"}],
    32: [{"title": "Correlation Heatmap (Numeric)", "category": "physicals"}],
    33: [{"title": "Top Speed Distribution", "category": "physicals"}],
    34: [{"title": "Stamina Score by Position", "category": "physicals"}],
    35: [{"title": "Offensive Contribution by Position", "category": "physicals"}],
    36: [{"title": "Defensive Contribution by Position", "category": "physicals"}],
    37: [{"title": "Top Players by Tournament Rating", "category": "physicals"}],
    38: [{"title": "Core Feature Pairplot", "category": "physicals"}],
    49: [
        {"title": "Confusion Matrix (Logistic Regression)", "category": "model"},
        {"title": "Confusion Matrix (Decision Tree)", "category": "model"},
        {"title": "Confusion Matrix (Random Forest)", "category": "model"},
        {"title": "Confusion Matrix (Gradient Boosting)", "category": "model"}
    ],
    50: [{"title": "Model Accuracy Comparison", "category": "model"}],
    52: [{"title": "Multiclass ROC Curve", "category": "model"}],
    53: [{"title": "Top 20 Important Features", "category": "model"}],
}

visualizations = []
for idx, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        if idx in image_mapping:
            maps = image_mapping[idx]
            img_outputs = [o for o in cell.get('outputs', []) if 'data' in o and 'image/png' in o['data']]
            for i, o in enumerate(img_outputs):
                if i < len(maps):
                    visualizations.append({
                        "title": maps[i]["title"],
                        "category": maps[i]["category"],
                        "image": o['data']['image/png']
                    })

print(f"Mapped {len(visualizations)} visualizations.")

print("Generating HTML...")
html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,700;12..96,800&display=swap" rel="stylesheet">
<style>
  html, body {{ margin: 0; height: 100%; }}
  body {{ background: #f3f5f2; }}
  
  @keyframes slideUpFade {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}
  
  .dashboard-grid {{
    display: grid; 
    grid-template-columns: repeat(4, 1fr); 
    grid-auto-rows: minmax(160px, auto); 
    gap: 18px; 
  }}
  
  .card {{
    background: #fff; 
    border-radius: 24px; 
    padding: 24px; 
    display: flex; 
    flex-direction: column;
    box-shadow: 0 4px 16px rgba(0,0,0,0.02);
    transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    opacity: 0;
    animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
  }}
  
  .card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 16px 36px rgba(0,0,0,0.06);
  }}
  
  .card:nth-child(1) {{ animation-delay: 0.05s; }}
  .card:nth-child(2) {{ animation-delay: 0.1s; }}
  .card:nth-child(3) {{ animation-delay: 0.15s; }}
  .card:nth-child(4) {{ animation-delay: 0.2s; }}
  .card:nth-child(5) {{ animation-delay: 0.25s; }}
  .card:nth-child(6) {{ animation-delay: 0.3s; }}
  .card:nth-child(7) {{ animation-delay: 0.35s; }}
  
  .card-dark {{ background: #111311; color: #fff; }}
  
  /* Tabs Styling */
  .tabs-container {{
      margin-top: 40px;
      animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
      animation-delay: 0.4s;
      opacity: 0;
  }}
  .tabs-header {{
      display: flex;
      gap: 8px;
      border-bottom: 2px solid rgba(0,0,0,0.04);
      padding-bottom: 12px;
      margin-bottom: 24px;
      overflow-x: auto;
  }}
  .tab-btn {{
      background: none;
      border: none;
      font-family: 'Bricolage Grotesque', sans-serif;
      font-size: 15px;
      font-weight: 600;
      color: #7a8078;
      padding: 10px 20px;
      cursor: pointer;
      border-radius: 12px;
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
      white-space: nowrap;
  }}
  .tab-btn:hover {{
      color: #1a1d1a;
      background: rgba(0,0,0,0.03);
  }}
  .tab-btn.active {{
      color: #059669;
      background: #e6f7f0;
  }}
  
  .tab-content {{
      display: none;
      grid-template-columns: repeat(3, 1fr);
      gap: 20px;
      opacity: 0;
      transform: translateY(12px);
      transition: opacity 0.3s ease, transform 0.3s ease;
  }}
  .tab-content.active {{
      display: grid;
      opacity: 1;
      transform: translateY(0);
  }}
  
  .image-card {{
      background: #fff;
      border-radius: 24px;
      padding: 20px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.02);
      display: flex;
      flex-direction: column;
      gap: 12px;
      cursor: pointer;
      transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.3s cubic-bezier(0.16, 1, 0.3, 1);
  }}
  .image-card:hover {{
      transform: translateY(-4px);
      box-shadow: 0 16px 36px rgba(0,0,0,0.06);
  }}
  .image-card img {{
      max-width: 100%;
      height: auto;
      border-radius: 14px;
  }}
  .image-title {{
      font-family: 'Bricolage Grotesque', sans-serif;
      font-weight: 600;
      font-size: 14px;
      color: #1a1d1a;
      text-align: center;
  }}
  
  /* Lightbox Modal */
  .lightbox {{
      display: none;
      position: fixed;
      z-index: 9999;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      background-color: rgba(17, 19, 17, 0.96);
      align-items: center;
      justify-content: center;
      flex-direction: column;
      opacity: 0;
      transition: opacity 0.3s ease;
  }}
  .lightbox.active {{
      display: flex;
      opacity: 1;
  }}
  .lightbox img {{
      max-width: 85%;
      max-height: 80%;
      border-radius: 16px;
      box-shadow: 0 24px 64px rgba(0,0,0,0.4);
      transform: scale(0.95);
      transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
  }}
  .lightbox.active img {{
      transform: scale(1);
  }}
  .close-btn {{
      position: absolute;
      top: 24px;
      right: 32px;
      color: #fff;
      font-size: 44px;
      font-weight: bold;
      cursor: pointer;
      transition: color 0.2s ease;
  }}
  .close-btn:hover {{
      color: #10b981;
  }}
  .lightbox-caption {{
      color: #fff;
      font-size: 18px;
      font-family: 'Bricolage Grotesque', sans-serif;
      margin-top: 18px;
      font-weight: 600;
      letter-spacing: -0.01em;
  }}
</style>
</head>
<body>
<div style="background:#f3f5f2; min-height:100vh; box-sizing:border-box; font-family:'Plus Jakarta Sans',sans-serif; color:#1a1d1a; padding:32px; display:flex; flex-direction:column; gap:24px;">

  <header style="display:flex; align-items:center; justify-content:space-between; animation: slideUpFade 0.5s ease forwards; border-bottom: 1px solid rgba(0,0,0,0.05); padding-bottom: 16px; margin-bottom: 8px;">
    <div style="display:flex; align-items:center; gap:14px;">
      <div style="width:38px; height:38px; border-radius:12px; background:linear-gradient(135deg, #10b981, #059669); display:flex; align-items:center; justify-content:center; box-shadow: 0 4px 12px rgba(16,185,129,0.25);"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M4 16l5-6 4 3 7-9"/></svg></div>
      <div><div style="font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:22px; letter-spacing:-.03em;">Forecast Studio</div><div style="font-size:12.5px; color:#7a8078;">Interactive Dashboard & Analytics Platform</div></div>
    </div>
  </header>

  <div class="dashboard-grid">
    
    <!-- Match Outcomes (Donut) -->
    <div class="card card-dark" style="grid-column: span 1; grid-row: span 2; justify-content: space-between;">
      <div style="font-size:13px; color:#b9c0b6; font-weight: 600;">Match Outcomes</div>
      <div style="flex:1; position:relative; min-height: 150px; margin: 12px 0;">
        <canvas id="outcomesChart"></canvas>
      </div>
      <div style="border-top: 1px solid rgba(255,255,255,0.08); padding-top: 12px; display: flex; justify-content: space-between; font-size: 11.5px; color: #8d948a;">
        <span>Tournament split</span>
        <span style="color:#10b981; font-weight: 600;">Live data</span>
      </div>
    </div>

    <!-- Positional Archetypes (Radar Chart) -->
    <div class="card" style="grid-column: span 3; grid-row: span 2;">
      <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div><div style="font-family:'Bricolage Grotesque',sans-serif; font-size:17px; font-weight:600;">Positional Archetypes</div><div style="font-size:12.5px; color:#7a8078; margin-top:2px;">Normalized attributes by position</div></div>
      </div>
      <div style="flex:1; position:relative; margin-top:12px; min-height:220px;">
          <canvas id="radarChart"></canvas>
      </div>
    </div>

    <!-- Goals vs Assists Bubble Chart -->
    <div class="card" style="grid-column: span 2; grid-row: span 2;">
      <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:15px; font-weight:600; margin-bottom:16px;">Offensive Output (Size = Rating)</div>
      <div style="flex:1; position:relative; min-height:250px;">
          <canvas id="bubbleChart"></canvas>
      </div>
    </div>

    <!-- Top Players Stacked Bar -->
    <div class="card" style="grid-column: span 2; grid-row: span 2;">
      <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:15px; font-weight:600; margin-bottom:16px;">Elite Performers (Offense vs Defense)</div>
      <div style="flex:1; position:relative; min-height:250px;">
          <canvas id="playersChart"></canvas>
      </div>
    </div>

    <!-- Age vs Rating Trend -->
    <div class="card" style="grid-column: span 2; grid-row: span 2;">
      <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:15px; font-weight:600; margin-bottom:16px;">Performance Curve by Age</div>
      <div style="height:200px; position:relative;">
          <canvas id="ageChart"></canvas>
      </div>
    </div>

    <!-- Random Forest Importances -->
    <div class="card" style="grid-column: span 2; grid-row: span 2;">
      <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:15px; font-weight:600; margin-bottom:16px;">Model Key Drivers (Random Forest)</div>
      <div style="height:200px; position:relative;">
          <canvas id="rfChart"></canvas>
      </div>
    </div>

  </div>
  
  <!-- TABBED IMAGE GALLERY SECTION -->
  <div class="tabs-container">
      <div style="display: flex; flex-direction: column; gap: 4px; margin-bottom: 20px;">
          <div style="font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:24px; letter-spacing:-.03em;">Deep Notebook Insights</div>
          <div style="font-size:14px; color:#7a8078;">Browse all 32 analysis visualizations directly compiled from your model training & exploratory research. Click any card to expand.</div>
      </div>
      
      <div class="tabs-header">
          <button class="tab-btn active" onclick="openTab(event, 'tab-demographics')">Demographics & Basics</button>
          <button class="tab-btn" onclick="openTab(event, 'tab-finances')">Finances & Scoring</button>
          <button class="tab-btn" onclick="openTab(event, 'tab-physicals')">Physicals & Contributions</button>
          <button class="tab-btn" onclick="openTab(event, 'tab-model')">Machine Learning Metrics</button>
      </div>
      
      <!-- DEMOGRAPHICS TAB -->
      <div id="tab-demographics" class="tab-content active">"""

for vis in visualizations:
    if vis["category"] == "demographics":
        html_content += f"""
          <div class="image-card" onclick="openLightbox(this.querySelector('img').src, '{vis['title']}')">
              <div class="image-title">{vis['title']}</div>
              <img src="data:image/png;base64,{vis['image']}" alt="{vis['title']}">
          </div>"""

html_content += """
      </div>
      
      <!-- FINANCES TAB -->
      <div id="tab-finances" class="tab-content">"""

for vis in visualizations:
    if vis["category"] == "finances":
        html_content += f"""
          <div class="image-card" onclick="openLightbox(this.querySelector('img').src, '{vis['title']}')">
              <div class="image-title">{vis['title']}</div>
              <img src="data:image/png;base64,{vis['image']}" alt="{vis['title']}">
          </div>"""

html_content += """
      </div>
      
      <!-- PHYSICALS TAB -->
      <div id="tab-physicals" class="tab-content">"""

for vis in visualizations:
    if vis["category"] == "physicals":
        # Make the pairplot and heatmap look wider
        span_style = "grid-column: span 2;" if "Pairplot" in vis["title"] or "Correlation" in vis["title"] else ""
        html_content += f"""
          <div class="image-card" style="{span_style}" onclick="openLightbox(this.querySelector('img').src, '{vis['title']}')">
              <div class="image-title">{vis['title']}</div>
              <img src="data:image/png;base64,{vis['image']}" alt="{vis['title']}">
          </div>"""

html_content += """
      </div>
      
      <!-- MODEL TAB -->
      <div id="tab-model" class="tab-content">"""

for vis in visualizations:
    if vis["category"] == "model":
        html_content += f"""
          <div class="image-card" onclick="openLightbox(this.querySelector('img').src, '{vis['title']}')">
              <div class="image-title">{vis['title']}</div>
              <img src="data:image/png;base64,{vis['image']}" alt="{vis['title']}">
          </div>"""

html_content += f"""
      </div>
  </div>
</div>

<!-- Lightbox Modal -->
<div id="lightbox" class="lightbox" onclick="closeLightbox()">
    <span class="close-btn">&times;</span>
    <img id="lightbox-img" src="" alt="Enlarged Plot">
    <div id="lightbox-caption" class="lightbox-caption"></div>
</div>

<script>
// Lightbox functions
function openLightbox(imgSrc, captionText) {{
    const lightbox = document.getElementById('lightbox');
    const img = document.getElementById('lightbox-img');
    const caption = document.getElementById('lightbox-caption');
    img.src = imgSrc;
    caption.innerText = captionText;
    lightbox.style.display = 'flex';
    setTimeout(() => {{
        lightbox.classList.add('active');
    }}, 10);
}}

function closeLightbox() {{
    const lightbox = document.getElementById('lightbox');
    lightbox.classList.remove('active');
    setTimeout(() => {{
        lightbox.style.display = 'none';
    }}, 300);
}}

// Tab navigation functions
function openTab(evt, tabId) {{
    // Hide all tab contents
    const contents = document.querySelectorAll('.tab-content');
    contents.forEach(content => {{
        content.classList.remove('active');
        setTimeout(() => {{
            if (!content.classList.contains('active')) {{
                content.style.display = 'none';
            }}
        }}, 300);
    }});
    
    // Deactivate all buttons
    const buttons = document.querySelectorAll('.tab-btn');
    buttons.forEach(btn => btn.classList.remove('active'));
    
    // Show active tab content
    const activeContent = document.getElementById(tabId);
    activeContent.style.display = 'grid';
    setTimeout(() => {{
        activeContent.classList.add('active');
    }}, 10);
    
    // Add active class to clicked button
    evt.currentTarget.classList.add('active');
}}

// Chart.js Setup
Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
Chart.defaults.color = "#7a8078";
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(26, 29, 26, 0.95)';
Chart.defaults.plugins.tooltip.titleFont = {{ family: 'Bricolage Grotesque', size: 14, weight: 'bold' }};
Chart.defaults.plugins.tooltip.bodyFont = {{ family: 'Plus Jakarta Sans', size: 13 }};
Chart.defaults.plugins.tooltip.padding = 12;
Chart.defaults.plugins.tooltip.cornerRadius = 8;
Chart.defaults.animation.duration = 1200;
Chart.defaults.animation.easing = 'easeOutQuart';

document.addEventListener('DOMContentLoaded', function() {{

    // 1. Match Outcomes
    new Chart(document.getElementById('outcomesChart'), {{
        type: 'doughnut',
        data: {{
            labels: {list(data['match_results'].keys())},
            datasets: [{{
                data: {list(data['match_results'].values())},
                backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                borderWidth: 0,
                hoverOffset: 6
            }}]
        }},
        options: {{ 
            responsive: true, 
            maintainAspectRatio: false, 
            cutout: '72%',
            plugins: {{ 
                legend: {{ 
                    position: 'bottom', 
                    labels: {{ color: '#fff', usePointStyle: true, boxWidth: 6, padding: 12, font: {{ size: 11 }} }} 
                }} 
            }} 
        }}
    }});

    // 2. Radar Chart
    const radarCtx = document.getElementById('radarChart').getContext('2d');
    const radarColors = ['rgba(16, 185, 129, 0.4)', 'rgba(139, 92, 246, 0.4)', 'rgba(245, 158, 11, 0.4)'];
    const radarBorders = ['#10b981', '#8b5cf6', '#f59e0b'];
    const rData = {json.dumps(data['radar_datasets'])};
    rData.forEach((ds, i) => {{
        ds.backgroundColor = radarColors[i];
        ds.borderColor = radarBorders[i];
        ds.pointBackgroundColor = radarBorders[i];
        ds.borderWidth = 2;
    }});

    new Chart(radarCtx, {{
        type: 'radar',
        data: {{
            labels: {json.dumps(data['radar_labels'])},
            datasets: rData
        }},
        options: {{
            responsive: true, maintainAspectRatio: false,
            scales: {{
                r: {{
                    angleLines: {{ color: 'rgba(0,0,0,0.05)' }},
                    grid: {{ color: 'rgba(0,0,0,0.05)' }},
                    pointLabels: {{ font: {{ family: 'Bricolage Grotesque', size: 13, weight: '600' }}, color: '#1a1d1a' }},
                    ticks: {{ display: false, min: 0, max: 100 }}
                }}
            }},
            plugins: {{ legend: {{ position: 'right', labels: {{ usePointStyle: true, boxWidth: 8 }} }} }}
        }}
    }});

    // 3. Bubble Chart
    const gaData = {json.dumps(data['ga_data'])};
    const colorsObj = {{'Forward': '#10b981', 'Midfielder': '#8b5cf6', 'Defender': '#f59e0b', 'Goalkeeper': '#ef4444'}};
    const bDatasets = Object.keys(colorsObj).map(pos => ({{
        label: pos,
        backgroundColor: colorsObj[pos] + 'aa',
        borderColor: colorsObj[pos],
        borderWidth: 1,
        data: gaData.filter(d => d.position === pos).map(d => ({{x: d.goals, y: d.assists, r: d.r, name: d.player_name, rating: d.tournament_rating}}))
    }}));
    new Chart(document.getElementById('bubbleChart'), {{
        type: 'bubble',
        data: {{ datasets: bDatasets }},
        options: {{
            responsive: true, maintainAspectRatio: false,
            plugins: {{
                tooltip: {{ 
                    callbacks: {{ 
                        title: (ctx) => ctx[0].raw.name,
                        label: (ctx) => `Goals: ${{ctx.raw.x}} | Assists: ${{ctx.raw.y}} | Rating: ${{ctx.raw.rating}}`
                    }} 
                }},
                legend: {{ position: 'bottom', labels: {{ usePointStyle: true, boxWidth: 8 }} }}
            }},
            scales: {{ 
                x: {{ grid: {{ color: 'rgba(0,0,0,0.03)' }}, title: {{ display: true, text: 'Total Goals', color: '#7a8078' }} }}, 
                y: {{ grid: {{ color: 'rgba(0,0,0,0.03)' }}, title: {{ display: true, text: 'Total Assists', color: '#7a8078' }} }} 
            }}
        }}
    }});

    // 4. Top Players Stacked
    new Chart(document.getElementById('playersChart'), {{
        type: 'bar',
        data: {{
            labels: {json.dumps(data['top_names'])},
            datasets: [
                {{ label: 'Offensive Contribution', data: {json.dumps(data['top_off'])}, backgroundColor: '#8b5cf6', borderRadius: 4 }},
                {{ label: 'Defensive Contribution', data: {json.dumps(data['top_def'])}, backgroundColor: '#10b981', borderRadius: 4 }}
            ]
        }},
        options: {{
            indexAxis: 'y',
            responsive: true, maintainAspectRatio: false,
            scales: {{ x: {{ stacked: true, display: false }}, y: {{ stacked: true, grid: {{ display: false }} }} }},
            plugins: {{ legend: {{ position: 'bottom', labels: {{ usePointStyle: true, boxWidth: 8 }} }} }}
        }}
    }});

    // 5. Age Trend Line Chart
    const ctxAge = document.getElementById('ageChart').getContext('2d');
    let gradAge = ctxAge.createLinearGradient(0, 0, 0, 200);
    gradAge.addColorStop(0, 'rgba(245, 158, 11, 0.4)');
    gradAge.addColorStop(1, 'rgba(245, 158, 11, 0)');
    new Chart(ctxAge, {{
        type: 'line',
        data: {{
            labels: {json.dumps(data['age_labels'])},
            datasets: [{{
                label: 'Avg Rating',
                data: {json.dumps(data['age_ratings'])},
                borderColor: '#f59e0b',
                backgroundColor: gradAge,
                fill: true,
                tension: 0.4,
                borderWidth: 3,
                pointRadius: 3
            }}]
        }},
        options: {{
            responsive: true, maintainAspectRatio: false,
            scales: {{ x: {{ grid: {{ display: false }}, title: {{ display: true, text: 'Player Age' }} }}, y: {{ grid: {{ color: 'rgba(0,0,0,0.03)' }}, title: {{ display: true, text: 'Avg Rating' }} }} }},
            plugins: {{ legend: {{ display: false }} }}
        }}
    }});

    // 6. Random Forest Key Drivers
    const ctxRF = document.getElementById('rfChart').getContext('2d');
    let gradRF = ctxRF.createLinearGradient(0, 0, 400, 0);
    gradRF.addColorStop(0, '#0284c7');
    gradRF.addColorStop(1, '#38bdf8');
    new Chart(ctxRF, {{
        type: 'bar',
        data: {{
            labels: {json.dumps(data['rf_features'])},
            datasets: [{{
                label: 'Importance %',
                data: {json.dumps(data['rf_importances'])},
                backgroundColor: gradRF,
                borderRadius: 4
            }}]
        }},
        options: {{
            indexAxis: 'y',
            responsive: true, maintainAspectRatio: false,
            scales: {{ x: {{ grid: {{ display: false }}, max: 100 }}, y: {{ grid: {{ display: false }} }} }},
            plugins: {{ legend: {{ display: false }} }}
        }}
    }});

    // Initialize display states for tab contents
    document.querySelectorAll('.tab-content').forEach(c => {{
        if(!c.classList.contains('active')) {{
            c.style.display = 'none';
        }}
    }});
}});
</script>
</body>
</html>
"""

with open('/Volumes/hard/fifa-data-visualizations/fifa.html', 'w') as f:
    f.write(html_content)

print("Super-Polished Interactive HTML Dashboard Generated Successfully!")
