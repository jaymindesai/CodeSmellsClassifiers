import context
import pandas

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

for file in context.FILES:

    # Retrieve the file name for reporting
    file_name = file.split('/')[-1]

    # Generate a dataframe from the csv file
    # Drop the two symbolic columns if present
    data_frame = pandas.read_csv(file)
    data_frame.drop(columns=context.SYM_COLS, inplace=True, errors='ignore')

    # Store the labels, translating True -> 1 and False -> 0
    # Remove the labels from the dataframe
    labels = data_frame['SMELLS'].apply(lambda x: 1 if x else 0)
    unlabeled_data = data_frame.drop(columns=['SMELLS'])

    # Initialize the stratified k-folds
    skfolds = StratifiedKFold(n_splits=5, random_state=0)

    # Initialize the ZeroR and Random Guess classifiers
    zero_r = DummyClassifier(strategy='most_frequent', random_state=0)
    rand_guess = DummyClassifier(strategy='uniform', random_state=0)

    # Set up a dictionary to record performance metrics for each run and fold
    metrics = {'f_score': [], 'kappa': [], 'pct_dth': [], 'inform': []}

    # FOR TESTING ONLY
    rand_metrics = {'f_score': [], 'kappa': [], 'pct_dth': [], 'inform': []}

    # Run the 5-fold cross-validation 5 times (25 results total)
    run = 0
    for i in range(5):
        fold = 0

        for train_indices, test_indices in skfolds.split(unlabeled_data, labels):

            # Clone the ZeroR and Random Guess classifiers
            cloned_zero_r = clone(zero_r)
            cloned_rand_guess = clone(rand_guess)

            # Retrieve the training data and training labels for the fold
            train_data = unlabeled_data.iloc[train_indices]
            train_labels = labels.iloc[train_indices]

            # Retrieve the testing data and testing labels for the fold
            test_data = unlabeled_data.iloc[test_indices]
            test_labels = labels.iloc[test_indices]

            # Train the ZeroR and Random Guess classifiers for the fold
            cloned_zero_r.fit(train_data, train_labels)
            cloned_rand_guess.fit(train_data, train_labels)

            # Test the ZeroR and Random Guess classifiers for the fold
            zero_r_pred = cloned_zero_r.predict(test_data)
            rand_guess_pred = cloned_rand_guess.predict(test_data)

            # Generate the TN, FP, FN, TP metrics for the ZeroR and Random Guess classifiers
            tn, fp, fn, tp = confusion_matrix(test_labels, zero_r_pred).ravel()
            rand_tn, rand_fp, rand_fn, rand_tp = confusion_matrix(test_labels, rand_guess_pred).ravel()

            metrics['f_score'].append(context.f_score())
            metrics['kappa'].append(context.kappa())
            metrics['pct_dth'].append(context.pct_dth())
            metrics['inform'].append(context.inform())

            rand_metrics['f_score'].append(context.f_score())
            rand_metrics['kappa'].append(context.kappa())
            rand_metrics['pct_dth'].append(context.pct_dth())
            rand_metrics['inform'].append(context.inform())

            fold += 1

        run += 1

    print('----- ' + file_name + ' -----\n')

    for metric in metrics:
        print(metric, metrics[metric])