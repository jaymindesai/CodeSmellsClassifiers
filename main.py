import os
import shutil

output_directories = ['_data-class', '_feature-envy', '_god-class', '_long-method']

shutil.rmtree(f'./_output', ignore_errors=True)
os.mkdir('./_output')

for code_smell in output_directories:
    os.mkdir(f'./_output/{code_smell}')

from bl_clfs import bl

# ----------

# import context
# import pandas
# import time
#
# from skfeature.function.statistical_based import CFS
# from skfeature.function.wrapper import decision_tree_backward, decision_tree_forward, svm_backward, svm_forward
#
# files = ['_data/_data-class.csv'] #, '_data/_feature-envy.csv', '_data/_god-class.csv', '_data/_long-method.csv']
#
# for file in files:
#
#     data_frame = pandas.read_csv(file)
#     data_frame.drop(columns=context.SYM_COLS, inplace=True, errors='ignore')
#     labels = data_frame['SMELLS'].apply(lambda x: 1 if x else 0)
#     unlabeled_data = data_frame.drop(columns=['SMELLS'])
#
#     start_time = time.perf_counter()
#     cfs_feats = CFS.cfs(unlabeled_data.values, labels.values)
#     print('CFS', sorted(cfs_feats))
#     end_time = time.perf_counter()
#     print(end_time - start_time, '\n')
#
#     start_time = time.perf_counter()
#     dtb_feats = decision_tree_backward.decision_tree_backward(unlabeled_data.values, labels.values, int((len(unlabeled_data) - 1) ** 0.5))
#     print('DTB', sorted(dtb_feats))
#     end_time = time.perf_counter()
#     print(end_time - start_time, '\n')
#
#     start_time = time.perf_counter()
#     dtf_feats = decision_tree_forward.decision_tree_forward(unlabeled_data.values, labels.values, int((len(unlabeled_data) - 1) ** 0.5))
#     print('DTF', sorted(dtf_feats))
#     end_time = time.perf_counter()
#     print(end_time - start_time, '\n')
#
#     start_time = time.perf_counter()
#     svmb_feats = svm_backward.svm_backward(unlabeled_data.values, labels.values, int((len(unlabeled_data) - 1) ** 0.5))
#     print('SVMB', sorted(svmb_feats))
#     end_time = time.perf_counter()
#     print(end_time - start_time, '\n')
#
#     start_time = time.perf_counter()
#     svmf_feats = svm_forward.svm_forward(unlabeled_data.values, labels.values, int((len(unlabeled_data) - 1) ** 0.5))
#     print('SVMF', sorted(svmf_feats))
#     end_time = time.perf_counter()
#     print(end_time - start_time, '\n')
#
#     print()
