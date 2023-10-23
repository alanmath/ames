# import nbformat

# def extract_code_cells(notebook_path):
#     """
#     Extrai células de código de um notebook Jupyter
#     """
#     with open(notebook_path, 'r', encoding='utf-8') as file:
#         notebook = nbformat.read(file, as_version=4)
#         code_cells = [cell['source'] for cell in notebook.cells if cell.cell_type == 'code']
#         return code_cells

# def save_to_py(file_path, code_cells):
#     """
#     Salva as células de código em um arquivo .py
#     """
#     with open(file_path, 'w', encoding='utf-8') as file:
#         for cell in code_cells:
#             file.write(cell)
#             file.write('\n\n')

# notebook_paths = ['01_reading_raw_data.ipynb', '02_analysis_and_preprocessing.ipynb', '02.5_agora_vai.ipynb']
# output_file_path = 'combined_script.py'

# # Extrair células de código de cada notebook
# all_code_cells = []
# for path in notebook_paths:
#     all_code_cells.extend(extract_code_cells(path))

# # Salvar todas as células de código em um arquivo .py
# save_to_py(output_file_path, all_code_cells)

