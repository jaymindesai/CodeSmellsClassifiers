import context
import pandas

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

for file in context.FILES:

    file_name = file.split('/')[-1]

    data_frame = pandas.read_csv(file)
    data_frame.drop(columns=context.SYM_COLS, inplace=True, errors='ignore')

    labels = data_frame['SMELLS'].apply(lambda x: 1 if x else 0)
    unlabeled_data = data_frame.drop(columns=['SMELLS'])

    skfolds = StratifiedKFold(n_splits=5, random_state=0)
    dummy = DummyClassifier(random_state=0)

    fold = 0

    print('----- ' + file_name + ' -----\n')

    for train_index, test_index in skfolds.split(unlabeled_data, labels):

        cloned_dummy = clone(dummy)

        X_train_folds = unlabeled_data.iloc[train_index]
        y_train_folds = labels.iloc[train_index]

        X_test_folds = unlabeled_data.iloc[test_index]
        y_test_folds = labels.iloc[test_index]

        cloned_dummy.fit(X_train_folds, y_train_folds)

        dummy_pred = cloned_dummy.predict(X_test_folds)

        tn, fp, fn, tp = confusion_matrix(y_test_folds, dummy_pred).ravel()

        print(f'FOLD {fold}', '>>', '\tTN:', tn, '\tFP:', fp, '\tFN:', fn, '\tTP:', tp)

        fold += 1

    print()

