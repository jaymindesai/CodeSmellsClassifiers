import os
import shutil

for root in ['./__norm', './__smote', './__both']:
    shutil.rmtree(root, ignore_errors=True)
    os.mkdir(root)
    for smell in ['_data-class', '_feature-envy', '_god-class', '_long-method']:
        os.mkdir(f'{root}/{smell}')

# from bl_clfs import bl
# from cl_clfs import cl
from fs_clfs import fs
