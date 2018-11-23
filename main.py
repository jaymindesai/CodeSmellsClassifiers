import os
import shutil

output_directories = ['_data-class', '_feature-envy', '_god-class', '_long-method']

shutil.rmtree(f'./_output', ignore_errors=True)
os.mkdir('./_output')

for code_smell in output_directories:
    os.mkdir(f'./_output/{code_smell}')

from bl_clfs import cart
from bl_clfs import nb
from bl_clfs import oner
from bl_clfs import rf
from bl_clfs import svm

# from cl_clfs import cl_cart
# from cl_clfs import cl_nb
# from cl_clfs import cl_zero_r

# from fs_clfs import fs_cart
# from fs_clfs import fs_nb
# from fs_clfs import fs_zero_r

# import context
# import pandas
#
# from skfeature.function.statistical_based import CFS
# from skfeature.function.wrapper import decision_tree_backward, decision_tree_forward, svm_backward, svm_forward
#
# files = ['_data/_data-class.csv', '_data/_feature-envy.csv', '_data/_god-class.csv', '_data/_long-method.csv']
#
# for file in files:
#
#     data_frame = pandas.read_csv(file)
#     data_frame.drop(columns=context.SYM_COLS, inplace=True, errors='ignore')
#     labels = data_frame['SMELLS'].apply(lambda x: 1 if x else 0)
#     unlabeled_data = data_frame.drop(columns=['SMELLS'])
#
#     cfs_feats = CFS.cfs(unlabeled_data.values, labels.values)
#     print('CFS', sorted(cfs_feats))
#
#     dtb_feats = decision_tree_backward.decision_tree_backward(unlabeled_data.values, labels.values, int((len(unlabeled_data) - 1) ** 0.5))
#     print('DTB', sorted(dtb_feats))
#
#     dtf_feats = decision_tree_forward.decision_tree_forward(unlabeled_data.values, labels.values, int((len(unlabeled_data) - 1) ** 0.5))
#     print('DTF', sorted(dtf_feats))
#
#     svmb_feats = svm_backward.svm_backward(unlabeled_data.values, labels.values, int((len(unlabeled_data) - 1) ** 0.5))
#     print('SVMB', sorted(svmb_feats))
#
#     svmf_feats = svm_forward.svm_forward(unlabeled_data.values, labels.values, int((len(unlabeled_data) - 1) ** 0.5))
#     print('SVMF', sorted(svmf_feats))
#
#     print()

    # for selected_feature in selected_features:
    #
    #     print(list(unlabeled_data)[selected_feature])
