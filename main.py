import os
import shutil

output_directories = ['_data-class', '_feature-envy', '_god-class', '_long-method']

shutil.rmtree(f'./_output', ignore_errors=True)
os.mkdir('./_output')

for code_smell in output_directories:
    os.mkdir(f'./_output/{code_smell}')

# from bl_clfs import bl
# from cl_clfs import cl
from fs_clfs import fs
