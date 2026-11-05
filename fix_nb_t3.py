import json

nb_path = 'notebooks/pretrained_models.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

changes = []

def replace_in_cell(idx, old, new, desc):
    cell = nb['cells'][idx]
    source = ''.join(cell['source'])
    if old not in source:
        print(f'  WARNING: Cell {idx}: NOT FOUND: {desc}')
        return False
    cell['source'] = source.replace(old, new).splitlines(keepends=True)
    changes.append(f'Cell {idx}: {desc}')
    return True

# Cell 20
replace_in_cell(20, 'save_config(', 'utils.save_config(', 'save_config -> utils.save_config')
replace_in_cell(20, 'save_history(', 'utils.save_history(', 'save_history -> utils.save_history')
replace_in_cell(20, 'load_model(', 'utils.load_model(', 'load_model -> utils.load_model')

# Cell 23
replace_in_cell(23, 'save_config(', 'utils.save_config(', 'save_config -> utils.save_config')
replace_in_cell(23, 'save_history(', 'utils.save_history(', 'save_history -> utils.save_history')
replace_in_cell(23, 'load_model(', 'utils.load_model(', 'load_model -> utils.load_model')

# Cell 28
replace_in_cell(28, 'load_model(', 'utils.load_model(', 'load_model -> utils.load_model')

# Cell 29
replace_in_cell(29, 'save_config(', 'utils.save_config(', 'save_config -> utils.save_config')
replace_in_cell(29, 'save_history(', 'utils.save_history(', 'save_history -> utils.save_history')
replace_in_cell(29, 'load_model(', 'utils.load_model(', 'load_model -> utils.load_model')

print(f'Changes made: {len(changes)}')
for c in changes:
    print(f'  - {c}')

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')
print('Notebook saved.')
