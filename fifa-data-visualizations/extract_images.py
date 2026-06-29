import json
import base64
import os

with open('/Volumes/hard/fifa-data-visualizations/model.ipynb', 'r') as f:
    nb = json.load(f)

img_idx = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for output in cell.get('outputs', []):
            if 'data' in output and 'image/png' in output['data']:
                img_data = output['data']['image/png']
                with open(f'/Volumes/hard/fifa-data-visualizations/img_{img_idx}.png', 'wb') as img_f:
                    img_f.write(base64.b64decode(img_data))
                img_idx += 1

print(f"Extracted {img_idx} images.")
