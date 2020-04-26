import re
from functools import partial
from multiprocessing.pool import Pool

from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import mlxtend.frequent_patterns as ml
from pandas import DataFrame

csv_file = "Foodmart_2020_PD.csv"

stores = {
    'Deluxe Supermarkets': ['8', '12', '13', '17', '19', '21'],
    'Gourmet Supermarkets': ['4', '6']
}


def store_groups(csv_file, stores, count=False):
    store_trx = dict()
    store_db = dict()

    with open(csv_file, 'r') as file:
        lines = file.readlines()

    for line in lines:
        id = re.search('\d+|$', line).group()

        for groups in stores.keys():
            if id in stores[groups]:
                try:
                    store_trx[groups].append(line)
                except KeyError:
                    store_trx[groups] = [line]

    for store in store_trx.keys():
        store_db[store] = Pattern(lines=store_trx[store], count=count)

    return store_db


# Pattern Support Functions


def rm_exists(some_list, value):
    while True:
        try:
            some_list.remove(value)
        except ValueError:
            break

    return some_list


def is_int(string):
    """
    Return whether the string can be interpreted as a integer.

    :param string: str, string to check for integer
    """

    try:
        int(string)
        return True
    except ValueError:
        return False


def is_next_int(index, string):
    try:
        if is_int(string[index]) and is_int(string[index + 1]):
            return False
        elif is_int(string[index]):
            return True
        else:
            return False
    except IndexError:
        if is_int(string[index]):
            return True
        else:
            return False


def make_transaction(entry, count, **kwargs):
    transaction = []
    key = entry.split('=')
    stores = kwargs.get('stores')

    if key[0] == 'STORE_ID':
        if stores:
            inserts = 0
            for group in stores.keys():

                if key[1] in stores[group]:
                    transaction.append(group)
                    inserts += 1

            if inserts == 0:
                transaction.append('error')
    else:
        if count is True:
            temp = [key[0] for _ in range(int(key[1]))]
            transaction += temp
        else:
            transaction.append(key[0])

    return transaction


def fit_data(count, **kwargs):
    transactions = []
    tranx = partial(make_transaction, count=count)
    lines = None

    csv_file = kwargs.get('csv_path')

    if csv_file:
        with open(csv_file, 'r') as file:
            lines = file.readlines()
    else:
        lines = kwargs.get('lines')

    if lines:
        for line in lines:
            line = line.rstrip('\n')
            line = line.split(',')
            line = rm_exists(line, value='')

            transaction = []
            for i, entry in enumerate(line):

                if len(entry) > 11:
                    indices = [n + 1 for n in range(len(entry)) if is_next_int(n, entry)]

                    if len(indices) > 1:
                        indices.insert(0, 0)
                        entries = [entry[indices[x - 1]:indices[x]] for x in range(1, len(indices))]

                        for val in entries:
                            transaction += tranx(entry=val, **kwargs)
                    else:
                        transaction += tranx(entry=entry, **kwargs)

                else:
                    transaction += tranx(entry=entry, **kwargs)

            if 'error' not in transaction:
                transactions.append(transaction)

        return transactions

    else:
        raise ValueError('Data not provided')


def build_dataset(count, **kwargs):

    transactions = fit_data(count, **kwargs)
    tr_enc = TransactionEncoder()
    trans_array = tr_enc.fit(transactions).transform(transactions)

    return DataFrame(trans_array, columns=tr_enc.columns_)


class Pattern:

    def __init__(self, count=False, **kwargs):

        self.dataset = build_dataset(count, **kwargs)
        self.items_set = self.create_item_set(order=False, min_support=0)
        self.rules = None

    def create_item_set(self, algorithm='fpgrowth', min_support=0.5, length=[None, None], order=True, ascending=True,
                        inplace=False):

        equal_len = None
        min_len = None
        max_len = None

        if length[0] == length[1] and length[0] and length[1]:
            equal_len = length[0]

        else:
            if length[0]:
                min_len = length[0]
            if length[1]:
                max_len = length[1]

        try:
            temp = getattr(ml, algorithm)(df=self.dataset, min_support=min_support, use_colnames=True, max_len=max_len)
        except AttributeError:
            raise ValueError('Algorithm does not exist')

        if equal_len or min_len or order is True:
            temp['length'] = temp['itemsets'].apply(lambda x: len(x))

            if equal_len:
                temp = temp[temp['length'] == equal_len]
            elif min_len:
                temp = temp[temp['length'] >= min_len]

            if order is True:
                temp.sort_values('length', inplace=True, ascending=ascending)
                temp.reset_index(drop=True, inplace=True)

            temp.drop('length', axis=1, inplace=True)

        if inplace is True:
            self.items_set = temp

        return temp

    def create_association_rules(self, items_set=None, metrics={'confidence': 0.4, 'lift': 1.1}):

        if items_set is None:
            if self.items_set is not None:
                items_set = self.items_set
            else:
                raise ValueError('Frequent itemsets not created')

        keys = list(metrics.keys())

        temp = association_rules(items_set, metric=keys[0], min_threshold=metrics[keys[0]])

        if len(keys) > 1:
            for i in range(1, len(keys)):
                temp = temp[temp[keys[i]] >= metrics[keys[i]]]

        self.rules = temp
        return temp




test = Pattern(csv_path=csv_file)
print(test.dataset)
#test.create_item_set(min_support=0.001, order=True, length=[None, 3])
#print(test.items_set)
#print(len(test.items_set))
#test.create_association_rules()
#print(test.rules)
