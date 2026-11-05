import json

nb_path = 'notebooks/pretrained_models.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find cells with flat save_config/save_history/load_model calls
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    for pat in ['save_config(', 'save_history(', 'load_model(']:
        # Check if it's a flat call (not utils.save_config etc.)
        if pat in src and f'utils.{pat}' not in src:
            print(f'Cell {i}: flat call "{pat}" found')
