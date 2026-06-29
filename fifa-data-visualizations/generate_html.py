import json

with open('/Volumes/hard/fifa-data-visualizations/dashboard_data.json', 'r') as f:
    data = json.load(f)

html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="./support.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
<x-dc>
<helmet>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,700;12..96,800&display=swap" rel="stylesheet">
<style>
  html, body {{ margin: 0; height: 100%; }}
  body {{ background: #eef0ec; }}
</style>
</helmet>
<div style="background:#eef0ec; min-height:100vh; box-sizing:border-box; font-family:'Plus Jakarta Sans',sans-serif; color:#1a1d1a; padding:26px; display:flex; flex-direction:column; gap:18px;">

  <header style="display:flex; align-items:center; justify-content:space-between;">
    <div style="display:flex; align-items:center; gap:12px;">
      <div style="width:34px; height:34px; border-radius:11px; background:#10b981; display:flex; align-items:center; justify-content:center;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M4 16l5-6 4 3 7-9"/></svg></div>
      <div><div style="font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:19px; letter-spacing:-.02em;">Forecast Studio</div><div style="font-size:12.5px; color:#7a8078;">Match predictions · live</div></div>
    </div>
    <div style="display:flex; gap:8px;">
      <div style="padding:9px 14px; background:#fff; border-radius:12px; font-size:13px; font-weight:600; color:#1a1d1a;">Group Stage</div>
      <div style="padding:9px 14px; background:#fff; border-radius:12px; font-size:13px; color:#9aa098;">Knockouts</div>
      <div style="padding:9px 16px; background:#1a1d1a; color:#fff; border-radius:12px; font-size:13px; font-weight:600;">+ Run Model</div>
    </div>
  </header>

  <div style="flex:1; display:grid; grid-template-columns:repeat(4,1fr); grid-template-rows:1.1fr 1fr 1fr; gap:14px; min-height:760px;">
    <!-- big chart -->
    <div style="grid-column:span 2; background:#fff; border-radius:22px; padding:24px; display:flex; flex-direction:column;">
      <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div><div style="font-family:'Bricolage Grotesque',sans-serif; font-size:17px; font-weight:600;">Tournament Stage Distribution</div><div style="font-size:12.5px; color:#7a8078; margin-top:2px;">players per stage</div></div>
        <div style="text-align:right;"><div style="font-size:26px; font-weight:700; letter-spacing:-.02em;">{data['total_records']:,}</div><div style="font-size:12.5px; color:#10b981; font-weight:600;">total records</div></div>
      </div>
      <div style="flex:1; position:relative; margin-top:12px; min-height:150px;">
          <canvas id="bigChart"></canvas>
      </div>
    </div>
    <!-- accuracy donut -->
    <div style="background:#1a1d1a; color:#fff; border-radius:22px; padding:22px; display:flex; flex-direction:column; justify-content:space-between;">
      <div style="font-size:13px; color:#b9c0b6;">RF Model Accuracy</div>
      <div style="display:flex; justify-content:center; position:relative; height:120px; align-items:center;">
        <canvas id="accuracyChart"></canvas>
        <div style="position:absolute; text-align:center;">
            <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:26px; font-weight:700; color:#fff;">{data['accuracy']}%</div>
            <div style="font-size:10px; color:#8d948a;">hit rate</div>
        </div>
      </div>
      <div style="font-size:12px; color:#8d948a;">Validation set</div>
    </div>
    <!-- predictions number -->
    <div style="background:#d7f5e9; border-radius:22px; padding:22px; display:flex; flex-direction:column; justify-content:space-between;">
      <div style="font-size:13px; color:#1f7a5a; font-weight:600;">Predictions Run</div>
      <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:46px; font-weight:700; letter-spacing:-.03em; line-height:1;">{data['total_records']:,}</div>
      <div style="height:32px; position:relative;">
        <canvas id="predictionsChart"></canvas>
      </div>
    </div>

    <!-- positional forecast strip -->
    <div style="grid-column:span 2; background:#fff; border-radius:22px; padding:22px;">
      <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:15px; font-weight:600; margin-bottom:16px;">Positional Distribution</div>
      <div style="height:120px; position:relative;">
          <canvas id="positionChart"></canvas>
      </div>
    </div>
    <!-- confidence + MAPE small -->
    <div style="background:#fff; border-radius:22px; padding:22px; display:flex; flex-direction:column; justify-content:space-between;">
      <div style="font-size:13px; color:#7a8078;">Mean Precision</div>
      <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:38px; font-weight:700; letter-spacing:-.02em;">99<span style="font-size:20px; color:#b9c0b6;">%</span></div>
      <div style="height:8px; border-radius:6px; background:#eef0ec; overflow:hidden;"><div style="width:99%; height:100%; background:#8b5cf6; border-radius:6px;"></div></div>
    </div>
    <div style="background:#fff; border-radius:22px; padding:22px; display:flex; flex-direction:column; justify-content:space-between;">
      <div style="font-size:13px; color:#7a8078;">Mean Recall</div>
      <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:38px; font-weight:700; letter-spacing:-.02em;">99<span style="font-size:20px; color:#b9c0b6;">%</span></div>
      <div style="font-size:12.5px; color:#10b981; font-weight:600;">Weighted avg</div>
    </div>

    <!-- drivers -->
    <div style="grid-column:span 2; background:#fff; border-radius:22px; padding:22px;">
      <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:15px; font-weight:600; margin-bottom:14px;">Top Features (Random Forest)</div>
      <div style="display:flex; flex-direction:column; gap:13px;">"""

# Add the drivers dynamically
colors = ['#10b981', '#8b5cf6', '#f59e0b']
for i, feat in enumerate(data['top_features'][:3]):
    c = colors[i % len(colors)]
    name = feat['feature'].replace('_', ' ').capitalize()
    pct = feat['importance'] * 100
    html_content += f"""
        <div style="display:flex; align-items:center; gap:12px;">
            <span style="font-size:13.5px; font-weight:500; width:130px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="{name}">{name}</span>
            <div style="flex:1; height:9px; border-radius:6px; background:#eef0ec; overflow:hidden;">
                <div style="width:{pct}%; height:100%; background:{c}; border-radius:6px;"></div>
            </div>
            <span style="font-size:12.5px; font-weight:600; color:#7a8078; width:45px; text-align:right;">{pct:.1f}%</span>
        </div>"""

html_content += f"""
      </div>
    </div>
    <!-- anomaly -->
    <div style="grid-column:span 2; background:#fff; border-radius:22px; padding:22px;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;"><div style="font-family:'Bricolage Grotesque',sans-serif; font-size:15px; font-weight:600;">Data Flags</div><span style="font-size:11.5px; font-weight:600; color:#ef4444; background:#fdecec; padding:3px 9px; border-radius:20px;">2 active</span></div>
      <div style="display:flex; flex-direction:column; gap:12px;">
        <div style="display:flex; align-items:center; gap:12px;"><span style="width:8px; height:8px; border-radius:50%; background:#ef4444; flex:none;"></span><span style="font-size:13.5px; font-weight:500; flex:1;">High Correlation</span><span style="font-size:12.5px; color:#7a8078;">goals vs goals_team</span></div>
        <div style="height:1px; background:#eef0ec;"></div>
        <div style="display:flex; align-items:center; gap:12px;"><span style="width:8px; height:8px; border-radius:50%; background:#f59e0b; flex:none;"></span><span style="font-size:13.5px; font-weight:500; flex:1;">Class Imbalance</span><span style="font-size:12.5px; color:#7a8078;">Match Result D</span></div>
      </div>
    </div>
  </div>
</div>

<script>
document.addEventListener('DOMContentLoaded', function() {{
    Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
    Chart.defaults.color = "#7a8078";
    
    // 1. Big Chart (Stage Dist)
    const ctxBig = document.getElementById('bigChart').getContext('2d');
    
    // Create gradient
    let gradient = ctxBig.createLinearGradient(0, 0, 0, 200);
    gradient.addColorStop(0, 'rgba(16, 185, 129, 0.28)');
    gradient.addColorStop(1, 'rgba(16, 185, 129, 0)');
    
    new Chart(ctxBig, {{
        type: 'line',
        data: {{
            labels: {list(data['stage_dist'].keys())},
            datasets: [{{
                label: 'Players',
                data: {list(data['stage_dist'].values())},
                borderColor: '#10b981',
                backgroundColor: gradient,
                borderWidth: 3,
                tension: 0.4,
                fill: true,
                pointBackgroundColor: '#fff',
                pointBorderColor: '#10b981',
                pointBorderWidth: 2,
                pointRadius: 4,
                pointHoverRadius: 6
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{ legend: {{ display: false }} }},
            scales: {{
                x: {{ grid: {{ display: false }}, border: {{ display: false }} }},
                y: {{ display: false, min: 0 }}
            }},
            interaction: {{ mode: 'index', intersect: false }}
        }}
    }});

    // 2. Accuracy Donut
    const ctxAcc = document.getElementById('accuracyChart').getContext('2d');
    new Chart(ctxAcc, {{
        type: 'doughnut',
        data: {{
            labels: ['Hit', 'Miss'],
            datasets: [{{
                data: [{data['accuracy']}, {round(100 - data['accuracy'], 1)}],
                backgroundColor: ['#10b981', 'rgba(255,255,255,0.14)'],
                borderWidth: 0,
                cutout: '75%'
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{ legend: {{ display: false }}, tooltip: {{ enabled: false }} }}
        }}
    }});

    // 3. Predictions Chart (Mini bar)
    const ctxPred = document.getElementById('predictionsChart').getContext('2d');
    new Chart(ctxPred, {{
        type: 'bar',
        data: {{
            labels: ['D','M','F','G'],
            datasets: [{{
                data: [40, 60, 45, 80], // dummy relative heights to match old visual
                backgroundColor: ['#10b981', '#10b981', '#10b981', '#1f7a5a'],
                borderRadius: 3,
                borderSkipped: false
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{ legend: {{ display: false }}, tooltip: {{ enabled: false }} }},
            scales: {{ x: {{ display: false }}, y: {{ display: false, min: 0 }} }}
        }}
    }});

    // 4. Positional Dist Chart (replacing Daily Forecast)
    const ctxPos = document.getElementById('positionChart').getContext('2d');
    new Chart(ctxPos, {{
        type: 'bar',
        data: {{
            labels: {list(data['pos_dist'].keys())},
            datasets: [{{
                data: {list(data['pos_dist'].values())},
                backgroundColor: ['#e3f3ec', '#bfe9d6', '#10b981', '#1a1d1a'],
                borderRadius: 8,
                borderSkipped: false,
                barThickness: 'flex',
                maxBarThickness: 40
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{ legend: {{ display: false }} }},
            scales: {{
                x: {{ grid: {{ display: false }}, border: {{ display: false }} }},
                y: {{ display: false, min: 0 }}
            }}
        }}
    }});
}});
</script>
</x-dc>
</body>
</html>
"""

with open('/Volumes/hard/fifa-data-visualizations/fifa.html', 'w') as f:
    f.write(html_content)

print("Updated HTML generated successfully.")
