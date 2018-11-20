import os
import shutil

output_directories = ['_data-class', '_feature-envy', '_god-class', '_long-method']

for code_smell in output_directories:
    shutil.rmtree(f'./_output/{code_smell}')
    os.mkdir(f'./_output/{code_smell}')

from bl_clfs import cart
from bl_clfs import nb
from bl_clfs import zero_r

# from cl_clfs import cl_cart
# from cl_clfs import cl_nb
# from cl_clfs import cl_zero_r
#
# from fs_clfs import fs_cart
# from fs_clfs import fs_nb
# from fs_clfs import fs_zero_r