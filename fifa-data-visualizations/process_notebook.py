import json

with open('/Volumes/hard/fifa-data-visualizations/fifa.html', 'r') as f:
    html = f.read()

with open('/Volumes/hard/fifa-data-visualizations/model.ipynb', 'r') as f:
    nb = json.load(f)

images_html = []
texts_html = []

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for output in cell.get('outputs', []):
            if 'data' in output:
                if 'image/png' in output['data']:
                    # some base64 strings might have newlines in the notebook json
                    b64 = "".join(output['data']['image/png']) if isinstance(output['data']['image/png'], list) else output['data']['image/png']
                    images_html.append(f'<img src="data:image/png;base64,{b64.strip()}" style="width:100%; height:auto; border-radius:12px; margin-bottom:16px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);" />')
                
                # We can also get text/plain
                if 'text/plain' in output['data']:
                    txt = "".join(output['data']['text/plain']) if isinstance(output['data']['text/plain'], list) else output['data']['text/plain']
                    texts_html.append(f'<pre style="background:#f4f5f3; padding:12px; border-radius:8px; font-size:12px; overflow-x:auto; margin-bottom:16px;">{txt}</pre>')

            elif output.get('name') == 'stdout':
                txt = "".join(output['text']) if isinstance(output['text'], list) else output['text']
                texts_html.append(f'<pre style="background:#1a1d1a; color:#fff; padding:12px; border-radius:8px; font-size:12px; overflow-x:auto; margin-bottom:16px;">{txt}</pre>')

# Create a new section to append
new_section = f"""
    <!-- Model Analysis Section -->
    <div style="grid-column:span 4; background:#fff; border-radius:22px; padding:24px; margin-top:20px;">
      <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:20px; font-weight:700; margin-bottom:20px;">Model Visualizations & Analysis</div>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:24px;">
        <div>
          <div style="font-size:15px; font-weight:600; margin-bottom:14px;">Analysis Outputs</div>
          {''.join(texts_html)}
        </div>
        <div>
          <div style="font-size:15px; font-weight:600; margin-bottom:14px;">Visualizations</div>
          {''.join(images_html)}
        </div>
      </div>
    </div>
"""

# Insert before the closing div of the main container
# The main container closes at the end of the file.
insertion_point = html.rfind("</div>\n</x-dc>")
if insertion_point != -1:
    new_html = html[:insertion_point] + new_section + html[insertion_point:]
    with open('/Volumes/hard/fifa-data-visualizations/fifa.html', 'w') as f:
        f.write(new_html)
    print("Integration successful.")
else:
    print("Could not find insertion point.")
