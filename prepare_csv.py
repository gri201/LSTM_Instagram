import json
import csv
import random
from pprint import pprint

input = open("input2.txt")
data = input.read()

data = '[' + data + ']'
data = data.replace('}{', '}, {')

json_data = json.loads(data)

new_data = []
for user in json_data:
    user_data = {
        'ID' : user['ID'],
        'Description' : user['Bio'],
        'Target' : None
    }
    if not user_data['Description'] == '':
        new_data.append(user_data)

data_size = len(new_data)
train_data_size = int(data_size / 20)

random.shuffle(new_data)
train_data = new_data[:train_data_size]
test_data = new_data[train_data_size:]

with open('train_data.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=';')
    for row in train_data:
        writer.writerow(row.values())

with open('test_data.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=';')
    for row in test_data:
        writer.writerow(row.values())
