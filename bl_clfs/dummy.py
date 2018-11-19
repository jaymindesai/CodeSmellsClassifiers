import context
import pandas

from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

for file in context.FILES:

    file_name = file.split('/')[-1]

    data_frame = pandas.read_csv(file)
    data_frame.drop(columns=context.SYM_COLS, inplace=True, errors='ignore')

    labels = data_frame['SMELLS'].apply(lambda x: 1 if x else 0)
    unlabeled_data = data_frame.drop(columns=['SMELLS'])

    X_train, X_test, y_train, y_test = train_test_split(unlabeled_data,
                                                        labels,
                                                        stratify=labels,
                                                        test_size=0.25,
                                                        random_state=0)

    dummy = DummyClassifier(random_state=0)

    dummy.fit(X_train, y_train)

    dummy_pred = dummy.predict(X_test)

    dummy_cmat = confusion_matrix(y_test, dummy_pred)

    print(dummy_cmat, '\n---')
