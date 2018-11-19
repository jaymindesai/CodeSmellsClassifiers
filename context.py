import os

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = f'{ROOT}/_data'

FILES = [f'{DATA_PATH}/_data-class.csv',
         f'{DATA_PATH}/_feature-envy.csv',
         f'{DATA_PATH}/_god-class.csv',
         f'{DATA_PATH}/_long-method.csv']

# FILES = [f'{DATA_PATH}/_data-class.csv']

SYM_COLS = ['isStatic_method',
            'isStatic_type']


def metrics(tn, fp, fn, tp):
    return None
