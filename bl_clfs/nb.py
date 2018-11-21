import context
import pandas

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle

# print('\nNB\n')

for file in context.FILES:

    # Retrieve the file name for reporting
    file_name = file.split('/')[-1].split('.')[0]

    # Generate a DataFrame from the csv file
    # Drop the two symbolic columns if present
    data_frame = pandas.read_csv(file)
    data_frame.drop(columns=context.SYM_COLS, inplace=True, errors='ignore')

    # Initialize the stratified k-folds
    skfolds = StratifiedKFold(n_splits=5, random_state=0)

    # Initialize the Naive Bayes and Random Guess classifiers
    nb = GaussianNB()
    rand_guess = DummyClassifier(strategy='uniform', random_state=0)

    # Set up a dictionary to record performance metrics for each run and fold
    metrics = {'f_score': [], 'kappa': [], 'pct_dth': [], 'inform': []}

    # Run the 5-fold cross-validation for 5 runs, shuffling each run (25 results total)
    for run in range(5):

        # Shuffle the DataFrame
        data_frame = shuffle(data_frame, random_state=0)

        # Store the labels, translating True -> 1 and False -> 0
        labels = data_frame['SMELLS'].apply(lambda x: 1 if x else 0)

        # Remove the labels from the dataframe
        unlabeled_data = data_frame.drop(columns=['SMELLS'])

        fold = 0

        for train_indices, test_indices in skfolds.split(unlabeled_data, labels):

            # Clone the Naive Bayes and Random Guess classifiers
            cloned_nb = clone(nb)
            cloned_rand_guess = clone(rand_guess)

            # Retrieve the training data and training labels for the fold
            train_data = unlabeled_data.iloc[train_indices]
            train_labels = labels.iloc[train_indices]

            # Retrieve the testing data and testing labels for the fold
            test_data = unlabeled_data.iloc[test_indices]
            test_labels = labels.iloc[test_indices]

            # Train the Naive Bayes and Random Guess classifiers for the fold
            cloned_nb.fit(train_data, train_labels)
            cloned_rand_guess.fit(train_data, train_labels)

            # Test the Naive Bayes and Random Guess classifiers for the fold
            nb_pred = cloned_nb.predict(test_data)
            rand_guess_pred = cloned_rand_guess.predict(test_data)

            # Retrieve the TN, FP, FN, TP metrics for the Naive Bayes and Random Guess classifiers
            tn, fp, fn, tp = confusion_matrix(test_labels, nb_pred).ravel()
            rand_tn, rand_fp, rand_fn, rand_tp = confusion_matrix(test_labels, rand_guess_pred).ravel()

            # Calculate the F-Score, Kappa, Percent Distance to Heaven, and Informedness metrics
            metrics['f_score'].append(context.f_score(fp, fn, tp))
            metrics['kappa'].append(context.kappa(tn, fp, fn, tp, rand_tn, rand_fp, rand_fn, rand_tp))
            metrics['pct_dth'].append(context.pct_dth(tn, fp, fn, tp))
            metrics['inform'].append(context.inform(tn, fp, fn, tp))

            fold += 1

        run += 1

    # print(f'----- {file_name} -----\n')
    #
    # for metric in metrics:
    #     print(f'-- {metric} --\n')
    #     for index, value in enumerate(metrics[metric]):
    #         if index == len(metrics[metric]) - 1:
    #             print(value, end='')
    #         else:
    #             print(value, end=',')
    #     print('\n')

    for metric in metrics:
        with open(f'{context.ROOT}/_output/{file_name}/{file_name}-{metric}.txt', 'a+') as output_file:
            output_file.write('NB\n')
            for index, value in enumerate(metrics[metric]):
                if index == len(metrics[metric]) - 1:
                    output_file.write(f'{value}')
                else:
                    output_file.write(f'{value} ')
            output_file.write('\n\n')
