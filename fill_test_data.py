import csv

def save_csv(data, counter):
    with open('train_data.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        for row in data:
            writer.writerow(row.values())

    print('\n\n')
    print('Current saved row: ', counter)
    print('\n\n')

data = []
with open('train_data.csv', 'r', newline='') as f:
    reader = csv.reader(f, delimiter=';')
    for row in reader:
        user_data = {
            'ID' : row[0],
            'Description' : row[1],
            'Target' : row[2]
        }
        data.append(user_data)

counter = 0
for user in data:
    print('\n\n\n\n\n\n\n\n\n')
    if not user['Target']:
        key = input(user['Description'])
        if key == '1':
            target = True
        else:
            target = False

        user['Target'] = target

    counter = counter + 1
    if counter % 20 == 0:
        save_csv(data, counter)

save_csv(data, counter)
