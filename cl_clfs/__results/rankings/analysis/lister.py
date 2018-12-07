import csv

grid = []

index = {'oner-def': 0,
         'oner-def-clust': 1,
         'nb-def': 2,
         'nb-def-clust': 3,
         'svm-def': 10,
         'svm-def-clust': 11,
         'cart-def': 18,
         'cart-def-clust': 19,
         'rf-def': 26,
         'rf-def-clust': 27}

file_path = 'data/lm-pctdth.csv'

with open(file_path) as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        new_row = [item.replace(' ', '') for item in row[:4]]
        new_row[3] = new_row[3].split('(')[0]
        if index.get(new_row[1], None) is not None:
            new_row.append(index[new_row[1]])
            grid.append(new_row)

grid.sort(key=lambda x: x[4])

metric = file_path.split('/')[1].split('-')[1].split('.')[0]

if metric in ['acc', 'fscore']:
    worst_rank = 0
    for row in grid:
        if int(row[0]) > worst_rank:
            worst_rank = int(row[0])

    for row in grid:
        row[0] = worst_rank - int(row[0]) + 1

elif metric == 'pctdth':
    for row in grid:
        row[2] = round(1 - float(row[2]), 2)

# print('%+15s' % 'RANK')
# for row in grid:
#     print('%+15s'%(row[1]), '', row[0])

# print('\n%+15s' % 'MEDIAN')

clust = []
print(file_path.split('/')[1].split('.')[0])
print('')
print('UNCLUSTERED')
print('')
for row in grid:
    if 'clust' in row[1]:
        clust.append(row)
    else:
        # print('%+15s' % (row[1]), '', row[2], '', row[0])
        print(row[2])

print('')
print('CLUST')
print('')
for row in clust:
    # print('%+15s' % (row[1]), '', row[2], '', row[0])
    print(row[2])

# print('\n%+15s' % 'IQR')
# for row in grid:
#     print('%+15s' % (row[1]), '', row[3])
