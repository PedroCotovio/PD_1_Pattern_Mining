import re
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import mlxtend.frequent_patterns as ml
from pandas import DataFrame
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx

# User access Functions

def draw_graph(rules, rules_to_show):

    """
    From: https://intelligentonlinetools.com/blog/2018/02/10/how-to-create-data-visualization-for-association-rules-in-data-mining/
    """
    G1 = nx.DiGraph()

    color_map = []
    N = 50
    colors = np.random.rand(N)
    strs = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']

    for i in range(rules_to_show):
        G1.add_nodes_from(["R" + str(i)])

        for a in rules.iloc[i]['antecedants']:
            G1.add_nodes_from([a])

            G1.add_edge(a, "R" + str(i), color=colors[i], weight=2)

        for c in rules.iloc[i]['consequents']:
            G1.add_nodes_from()

            G1.add_edge("R" + str(i), c, color=colors[i], weight=2)

    for node in G1:
        found_a_string = False
        for item in strs:
            if node == item:
                found_a_string = True
        if found_a_string:
            color_map.append('yellow')
        else:
            color_map.append('green')

    edges = G1.edges()
    colors = [G1[u][v]['color'] for u, v in edges]
    weights = [G1[u][v]['weight'] for u, v in edges]

    pos = nx.spring_layout(G1, k=16, scale=1)
    nx.draw(G1, pos, edges=edges, node_color=color_map, edge_color=colors, width=weights, font_size=16,
            with_labels=False)

    for p in pos:  # raise text positions
        pos[p][1] += 0.07
    nx.draw_networkx_labels(G1, pos)
    plt.show()

def max_itemsets(frequent):

    """
    Basic algorithm to extract Maximal itemsets from frequent itemsets.

    :param frequent: frequent itemsets dataframe, should be ordered by length (descending)
    :return: Maximal itemset dataframe
    """

    itemsets = [set(item_set) for item_set in frequent['itemsets'].values]
    super_lenght = len(itemsets[0])
    deletion_rows = []
    super_sets = dict()

    for i, itemset in enumerate(itemsets):
        rerun = True
        while rerun is True:
            if len(itemset) == super_lenght:
                rerun = False

                try:
                    super_sets[super_lenght].append(itemset)
                except KeyError:
                    super_sets[super_lenght] = []
                    super_sets[super_lenght].append(itemset)

            elif len(itemset) == super_lenght - 1:
                rerun = False

                count = [sits for sits in super_sets[super_lenght] if itemset.issubset(sits)]
                if len(count) != 0:
                    deletion_rows.append(i)
                else:
                    try:
                        super_sets[super_lenght - 1].append(itemset)
                    except KeyError:
                        super_sets[super_lenght - 1] = []
                        super_sets[super_lenght - 1].append(itemset)

            elif len(itemset) == super_lenght - 2:
                rerun = True

                super_lenght -= 1

    frequent.drop(deletion_rows, axis=0, inplace=True)

    return frequent


def store_groups(csv_file, stores, count=False):

    """
    This function parses a csv file identifying store groups, and creating individual pattern objects for each group.

    :param csv_file: File to parse
    :param stores: Dict with store groups as keys and store id's as values
    :param count: If items quantities should be used
    :return: dict with store groups as keys, and pattern objects as values
    """

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


def get_n(values, keys, n, top=True):

    """
    Get N Top/Bottom items, from to ordered correspondent lists.

    :param values: List of values to parse
    :param keys: list of correspondent keys
    :param n: number of items to extract
    :param top: if top results should be used, else bottom results will be extracted
    :return: dict of N top/bottom items
    """

    top_items = []
    top_values = []
    for i in range(len(values)):
        top_items.append(keys[i])
        top_values.append(values[i])

        if top is True:
            if len(top_values) > n:
                index = top_values.index(min(top_values))
                del top_values[index]
                del top_items[index]
        else:
            if len(top_values) > n:
                index = top_values.index(max(top_values))
                del top_values[index]
                del top_items[index]

    return dict(zip(top_items, top_values))

# Class Support Functions


def rm_exists(some_list, value):

    """
    Delete all occurrences of var if exits on list

    :param some_list: list to parse
    :param value: value to delete
    :return: clean list
    """

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

    """
    Return whether a char in a string can be interpreted as the last value or unique char of an integer.

    :param index: int, char index
    :param string: str, string to check for integer

    """
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

    """
    This is a sub-function of 'worker' that parses a single entry.

    :param entry: string, un-parsed entry from csv file
    :param count: whether item quantities should be parsed
    :param stores: dict with store groups, if store group is to be considered inside the TRXN
    :return: list of parsed entry values to append to transaction
    """
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


def worker(line, tranx, **kwargs):
    """
    Worker function of 'fit_data' to parse single line of csv file

    :param line: csv file line as string
    :param tranx: entry parsing function (default=parcial(make_transaction))
    :param kwargs: optional args to pass on
    :return: list of transaction items
    """
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
        return transaction

# if __name__ == '__main__'


def fit_data(count, **kwargs):

    """
    Function for parallel parsing of transaction db as csv file or list of lines.

    :param csv_file: csv file path (optional), if not defined will be used.
    :param lines: list of file lines (optional), should be passed if csv_file is Null
    :param count: whether  items quantities should be used
    :param workers: number of workers to use, if not defined equals the number of available cpu's.
    :param kwargs: optional args to pass on
    :return: list of transactions
    """
    tranx = partial(make_transaction, count=count)
    lines = None

    csv_file = kwargs.get('csv_path')

    if csv_file:
        with open(csv_file, 'r') as file:
            lines = file.readlines()
    else:
        lines = kwargs.get('lines')

    if lines:
        workers = kwargs.get('workers')
        if workers:
            pool = Pool(workers)
        else:
            pool = Pool(cpu_count())

        transactions = pool.map(partial(worker, tranx=tranx, **kwargs), lines)
        return transactions

    else:
        raise ValueError('Data not provided')


def build_dataset(count, **kwargs):

    """
    Parse csv file or list of lines into a OneShot TRXN dataframe

    :param count: whether  items quantities should be used
    :return: OneShot encoded transactions dataframe
    """

    transactions = fit_data(count, **kwargs)
    tr_enc = TransactionEncoder()
    trans_array = tr_enc.fit(transactions).transform(transactions)

    return DataFrame(trans_array, columns=tr_enc.columns_)

# Pattern Class

class Pattern:

    """
    Pattern class is an api extension for pattern mining python libraries. That facilitates quick analyses,
    by providing an abstraction layer, with easy to handle objects. (It's a rudimentary implementation)

    It was designed to work with 'MLxtend', but can be easily expanded to work with other PM libs.
    """

    def __init__(self, count=False, **kwargs):

        """
        :param count: whether  items quantities should be used
        :param csv_file: csv file path (optional), if not defined will be used.
        :param lines: list of file lines (optional), should be passed if csv_file is Null
        :param workers: number of workers to use, if not defined equals the number of available cpu
        :param stores: dict with store groups, if store group is to be considered inside the TRXN
        """

        self.dataset = build_dataset(count, **kwargs)
        self.items_set = None
        self.rules = None

    def create_item_set(self, algorithm='fpgrowth', min_support=0.5, length=[None, None], order=True, ascending=True,
                        inplace=False):

        """
        Computes frequent itemsets from a one-hot DataFrame

        :param algorithm: implementation that should be used (fpgrowth, fpmax, apriori)
        :param min_support: Minimum support to consider, float [0:1]
        :param length: range of itemsets length as list in the format [min_length, max_length] ,
        if any is 'None' is not considered, to produce equalities list[0] == list[1].
        :param order: whether results should be ordered by length
        :param ascending: whether order should be ascending, else order descending
        :param inplace: whether frequent itemset should be saved to the object
        :return: frequent itemsets Dataframe
        """

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

            temp.drop('length', axis=1, inplace=True)
            temp.reset_index(drop=True, inplace=True)

        if inplace is True:
            self.items_set = temp

        return temp

    def create_association_rules(self, items_set=None, metrics={'confidence': 0.4, 'lift': 1.1}):

        """
        Computes a DataFrame of association rules including the metrics 'score', 'confidence', and 'lift'

        :param items_set: itemsets to use, if 'None', object sets are used.
        :param metrics: dict of metrics to filter rules.
        :return: DataFrame of association rules
        """

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

        temp.reset_index(drop=True, inplace=True)
        self.rules = temp
        return temp


# Testing #

#csv = "Foodmart_2020_PD.csv"

#stores = {
#    'Deluxe Supermarkets': ['8', '12', '13', '17', '19', '21'],
#    'Gourmet Supermarkets': ['4', '6']
#}

#test = Pattern(csv_path=csv, workers=5)
#print(test.dataset)
#test.create_item_set(min_support=0.001, order=True, length=[None, 3])
#print(test.items_set)
#print(len(test.items_set))
#test.create_association_rules()
#print(test.rules)
