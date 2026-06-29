import json

with open('/Volumes/hard/fifa-data-visualizations/dashboard_data.json', 'r') as f:
    data = json.load(f)

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
  body {{ background: #eef0ec; }}
  
  @keyframes slideUpFade {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}
  
  .dashboard-grid {{
    display: grid; 
    grid-template-columns: repeat(4, 1fr); 
    grid-auto-rows: minmax(160px, auto); 
    gap: 16px; 
    min-height: 760px;
  }}
  
  .card {{
    background: #fff; 
    border-radius: 22px; 
    padding: 24px; 
    display: flex; 
    flex-direction: column;
    box-shadow: 0 4px 12px rgba(0,0,0,0.02);
    transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    opacity: 0;
    animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
  }}
  
  .card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 12px 28px rgba(0,0,0,0.06);
  }}
  
  /* Stagger animations for cards */
  .card:nth-child(1) {{ animation-delay: 0.05s; }}
  .card:nth-child(2) {{ animation-delay: 0.1s; }}
  .card:nth-child(3) {{ animation-delay: 0.15s; }}
  .card:nth-child(4) {{ animation-delay: 0.2s; }}
  .card:nth-child(5) {{ animation-delay: 0.25s; }}
  .card:nth-child(6) {{ animation-delay: 0.3s; }}
  .card:nth-child(7) {{ animation-delay: 0.35s; }}
  .card:nth-child(8) {{ animation-delay: 0.4s; }}
  .card:nth-child(9) {{ animation-delay: 0.45s; }}
  
  .card-dark {{ background: #1a1d1a; color: #fff; }}
  .card-green {{ background: #d7f5e9; }}
</style>
</head>
<body>
<div style="background:#eef0ec; min-height:100vh; box-sizing:border-box; font-family:'Plus Jakarta Sans',sans-serif; color:#1a1d1a; padding:26px; display:flex; flex-direction:column; gap:18px;">

  <header style="display:flex; align-items:center; justify-content:space-between; animation: slideUpFade 0.5s ease forwards;">
    <div style="display:flex; align-items:center; gap:12px;">
      <div style="width:34px; height:34px; border-radius:11px; background:linear-gradient(135deg, #10b981, #059669); display:flex; align-items:center; justify-content:center; box-shadow: 0 4px 10px rgba(16,185,129,0.3);"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M4 16l5-6 4 3 7-9"/></svg></div>
      <div><div style="font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:19px; letter-spacing:-.02em;">Forecast Studio</div><div style="font-size:12.5px; color:#7a8078;">Match predictions · live</div></div>
    </div>
  </header>

  <div class="dashboard-grid">
    <!-- ROW 1 & 2 -->
    <div class="card" style="grid-column: span 2; grid-row: span 2;">
      <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div><div style="font-family:'Bricolage Grotesque',sans-serif; font-size:17px; font-weight:600;">Tournament Stage Distribution</div><div style="font-size:12.5px; color:#7a8078; margin-top:2px;">players per stage</div></div>
        <div style="text-align:right;"><div style="font-size:26px; font-weight:700; letter-spacing:-.02em;">{data['total_records']:,}</div><div style="font-size:12.5px; color:#10b981; font-weight:600;">total records</div></div>
      </div>
      <div style="flex:1; position:relative; margin-top:12px; min-height:200px;">
          <canvas id="bigChart"></canvas>
      </div>
    </div>

    <!-- Accuracy Donut -->
    <div class="card card-dark" style="justify-content:space-between; grid-row: span 2;">
      <div style="font-size:13px; color:#b9c0b6;">RF Model Accuracy</div>
      <div style="display:flex; justify-content:center; position:relative; height:140px; align-items:center;">
        <canvas id="accuracyChart"></canvas>
        <div style="position:absolute; text-align:center;">
            <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:26px; font-weight:700; color:#fff;">{data['accuracy']}%</div>
            <div style="font-size:10px; color:#8d948a;">hit rate</div>
        </div>
      </div>
      <div style="font-size:12px; color:#8d948a;">Validation set</div>
    </div>

    <!-- Match Outcomes -->
    <div class="card" style="justify-content:space-between; grid-row: span 2;">
      <div style="font-size:13px; color:#7a8078;">Match Outcomes</div>
      <div style="position:relative; height:140px;">
        <canvas id="outcomesChart"></canvas>
      </div>
      <div style="font-size:12px; color:#10b981; text-align:center; font-weight:600;">Win / Draw / Loss</div>
    </div>

    <!-- ROW 3 & 4 -->
    <div class="card" style="grid-column: span 2; grid-row: span 2;">
      <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:15px; font-weight:600; margin-bottom:16px;">Top Players (Tournament Rating)</div>
      <div style="flex:1; position:relative; min-height:250px;">
          <canvas id="playersChart"></canvas>
      </div>
    </div>

    <div class="card" style="grid-column: span 2; grid-row: span 2;">
      <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:15px; font-weight:600; margin-bottom:16px;">Goals vs Assists (Sample)</div>
      <div style="flex:1; position:relative; min-height:250px;">
          <canvas id="scatterChart"></canvas>
      </div>
    </div>

    <!-- ROW 5 -->
    <div class="card" style="grid-column: span 2;">
      <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:15px; font-weight:600; margin-bottom:16px;">Positional Distribution</div>
      <div style="height:120px; position:relative;">
          <canvas id="positionChart"></canvas>
      </div>
    </div>

    <div class="card" style="grid-column: span 2;">
      <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:15px; font-weight:600; margin-bottom:16px;">Average Stamina by Position</div>
      <div style="height:120px; position:relative;">
          <canvas id="staminaChart"></canvas>
      </div>
    </div>

    <!-- ROW 6 -->
    <div class="card" style="grid-column: span 2;">
      <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:15px; font-weight:600; margin-bottom:16px;">Top Speed Distribution (km/h)</div>
      <div style="height:120px; position:relative;">
          <canvas id="speedChart"></canvas>
      </div>
    </div>

    <!-- Drivers -->
    <div class="card" style="grid-column:span 2; justify-content:center;">
      <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:15px; font-weight:600; margin-bottom:14px;">Top Features (Random Forest)</div>
      <div style="display:flex; flex-direction:column; gap:13px;">"""

colors = ['#10b981', '#8b5cf6', '#f59e0b']
for i, feat in enumerate(data['top_features'][:3]):
    c = colors[i % len(colors)]
    name = feat['feature'].replace('_', ' ').capitalize()
    pct = feat['importance'] * 100
    html_content += f"""
        <div style="display:flex; align-items:center; gap:12px;">
            <span style="font-size:13.5px; font-weight:500; width:130px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="{name}">{name}</span>
            <div style="flex:1; height:9px; border-radius:6px; background:#eef0ec; overflow:hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);">
                <div style="width:{pct}%; height:100%; background:linear-gradient(90deg, {c}, {c}dd); border-radius:6px; transition: width 1s cubic-bezier(0.16, 1, 0.3, 1);"></div>
            </div>
            <span style="font-size:12.5px; font-weight:600; color:#7a8078; width:45px; text-align:right;">{pct:.1f}%</span>
        </div>"""

html_content += f"""
      </div>
    </div>

  </div>
</div>

<script>
// Base configuration for Chart.js
Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
Chart.defaults.color = "#7a8078";
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(26, 29, 26, 0.9)';
Chart.defaults.plugins.tooltip.titleFont = {{ family: 'Bricolage Grotesque', size: 14, weight: 'bold' }};
Chart.defaults.plugins.tooltip.bodyFont = {{ family: 'Plus Jakarta Sans', size: 13 }};
Chart.defaults.plugins.tooltip.padding = 12;
Chart.defaults.plugins.tooltip.cornerRadius = 8;
Chart.defaults.plugins.tooltip.boxPadding = 6;
Chart.defaults.animation.duration = 1000;
Chart.defaults.animation.easing = 'easeOutQuart';

document.addEventListener('DOMContentLoaded', function() {{
    
    // 1. Stage Dist
    const ctxBig = document.getElementById('bigChart').getContext('2d');
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
        options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }}, scales: {{ x: {{ grid: {{ display: false }} }}, y: {{ display: false }} }}, interaction: {{ mode: 'index', intersect: false }} }}
    }});

    // 2. Accuracy Donut
    new Chart(document.getElementById('accuracyChart'), {{
        type: 'doughnut',
        data: {{
            labels: ['Hit', 'Miss'],
            datasets: [{{
                data: [{data['accuracy']}, {round(100 - data['accuracy'], 1)}],
                backgroundColor: ['#10b981', 'rgba(255,255,255,0.08)'],
                hoverBackgroundColor: ['#059669', 'rgba(255,255,255,0.1)'],
                borderWidth: 0,
                cutout: '78%'
            }}]
        }},
        options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }}, tooltip: {{ enabled: true }} }} }}
    }});

    // 3. Match Outcomes
    new Chart(document.getElementById('outcomesChart'), {{
        type: 'doughnut',
        data: {{
            labels: {list(data['match_results'].keys())},
            datasets: [{{
                data: {list(data['match_results'].values())},
                backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                hoverOffset: 4,
                borderWidth: 2,
                borderColor: '#fff'
            }}]
        }},
        options: {{ responsive: true, maintainAspectRatio: false, cutout: '65%', plugins: {{ legend: {{ position: 'bottom', labels: {{ padding: 16, usePointStyle: true }} }} }} }}
    }});

    // 4. Top Players Leaderboard
    const ctxPlayers = document.getElementById('playersChart').getContext('2d');
    let gradPlayers = ctxPlayers.createLinearGradient(0, 0, 400, 0);
    gradPlayers.addColorStop(0, '#8b5cf6');
    gradPlayers.addColorStop(1, '#a78bfa');
    
    new Chart(ctxPlayers, {{
        type: 'bar',
        data: {{
            labels: {list(data['top_players'].keys())},
            datasets: [{{
                label: 'Rating',
                data: {list(data['top_players'].values())},
                backgroundColor: gradPlayers,
                borderRadius: 4,
                barThickness: 'flex',
                maxBarThickness: 16
            }}]
        }},
        options: {{
            indexAxis: 'y',
            responsive: true, maintainAspectRatio: false,
            plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: (ctx) => 'Avg Rating: ' + ctx.raw.toFixed(2) }} }} }},
            scales: {{ x: {{ display: false, min: 0, max: 10 }}, y: {{ grid: {{ display: false }} }} }}
        }}
    }});

    // 5. Goals vs Assists
    const gaData = {json.dumps(data['ga_data'])};
    const colorsObj = {{'Forward': '#10b981', 'Midfielder': '#8b5cf6', 'Defender': '#f59e0b', 'Goalkeeper': '#ef4444'}};
    const datasets = Object.keys(colorsObj).map(pos => ({{
        label: pos,
        backgroundColor: colorsObj[pos],
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: colorsObj[pos],
        pointHoverBorderWidth: 2,
        pointHoverRadius: 6,
        data: gaData.filter(d => d.position === pos).map(d => ({{x: d.goals, y: d.assists, name: d.player_name}}))
    }}));
    new Chart(document.getElementById('scatterChart'), {{
        type: 'scatter',
        data: {{ datasets: datasets }},
        options: {{
            responsive: true, maintainAspectRatio: false,
            plugins: {{
                tooltip: {{ 
                    callbacks: {{ 
                        title: (ctx) => ctx[0].raw.name,
                        label: (ctx) => `Goals: ${{ctx.raw.x}} | Assists: ${{ctx.raw.y}}`
                    }} 
                }}
            }},
            scales: {{ 
                x: {{ grid: {{ color: 'rgba(0,0,0,0.03)' }}, title: {{ display: true, text: 'Total Goals', color: '#b9c0b6' }} }}, 
                y: {{ grid: {{ color: 'rgba(0,0,0,0.03)' }}, title: {{ display: true, text: 'Total Assists', color: '#b9c0b6' }} }} 
            }}
        }}
    }});

    // 6. Positional Dist
    new Chart(document.getElementById('positionChart'), {{
        type: 'bar',
        data: {{
            labels: {list(data['pos_dist'].keys())},
            datasets: [{{
                data: {list(data['pos_dist'].values())},
                backgroundColor: ['#e3f3ec', '#bfe9d6', '#10b981', '#1a1d1a'],
                borderRadius: 8
            }}]
        }},
        options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }}, scales: {{ x: {{ grid: {{ display: false }} }}, y: {{ display: false }} }} }}
    }});

    // 7. Stamina by position
    const ctxStamina = document.getElementById('staminaChart').getContext('2d');
    let gradStamina = ctxStamina.createLinearGradient(0, 0, 0, 120);
    gradStamina.addColorStop(0, '#f59e0b');
    gradStamina.addColorStop(1, '#fbbf24');
    
    new Chart(ctxStamina, {{
        type: 'bar',
        data: {{
            labels: {list(data['stamina'].keys())},
            datasets: [{{
                data: {list(data['stamina'].values())},
                backgroundColor: gradStamina,
                borderRadius: 8
            }}]
        }},
        options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }}, scales: {{ x: {{ grid: {{ display: false }} }}, y: {{ display: false }} }} }}
    }});

    // 8. Speed Distribution
    const ctxSpeed = document.getElementById('speedChart').getContext('2d');
    let gradSpeed = ctxSpeed.createLinearGradient(0, 0, 0, 120);
    gradSpeed.addColorStop(0, 'rgba(139, 92, 246, 0.5)');
    gradSpeed.addColorStop(1, 'rgba(139, 92, 246, 0)');
    new Chart(ctxSpeed, {{
        type: 'line',
        data: {{
            labels: {data['speed_bins']},
            datasets: [{{
                label: 'Players',
                data: {data['speed_counts']},
                borderColor: '#8b5cf6',
                backgroundColor: gradSpeed,
                fill: true,
                tension: 0.4,
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 6,
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#8b5cf6',
                pointHoverBorderWidth: 2
            }}]
        }},
        options: {{ 
            responsive: true, maintainAspectRatio: false, 
            plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ title: (ctx) => ctx[0].label + ' km/h' }} }} }}, 
            scales: {{ x: {{ display: false }}, y: {{ display: false }} }},
            interaction: {{ mode: 'index', intersect: false }}
        }}
    }});
}});
</script>
</body>
</html>
"""

with open('/Volumes/hard/fifa-data-visualizations/fifa.html', 'w') as f:
    f.write(html_content)

print("Polished HTML Generated")
