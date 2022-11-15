"""
tex.load_filtered_data()

tx = tex.get_resulting_df(FileType.transactions)
txin = tex.get_resulting_df(FileType.inputs)
txout = tex.get_resulting_df(FileType.outputs)
"""


# input transakce zgroupnout
# na size ma vliv i to, kolik tam bylo inputu
# nezalezi na tom, kolik posle clovek, ale kolik tech lidi to posila
# cdd
# rozrezat si transakce na nekolik podskupin

# ===================================================================================================
# PREPROCESSING
# ===================================================================================================

# get times of the block
block_time_out = txout[['block_id', 'time']].drop_duplicates()
block_time_out = block_time_out.sort_values(by=['block_id'])
time_of_a_block = block_time_out['time'].tolist()

# filtered data (each adress is there only once per block)
data_out = txout.groupby('transaction_hash').max('value')
data_out = data_out[['block_id', 'value', 'value_usd', 'is_from_coinbase', 'is_spendable']]

# filtered fee data
data_tx = tx[['block_id', 'fee_per_kb']]
grouped_data_tx = data_tx.groupby('block_id').max('fee_per_kb')
indexed_data = grouped_data_tx.reset_index()
fee_per_kb = list(dict(indexed_data.values).values())

# sorted data_out by the block_id
sorted_data = data_out.sort_values(by=['block_id'])

# list of block ids
block_ids = list(dict(indexed_data.values).keys())

# number of transactions in a block
transaction_count = dict(sorted_data['block_id'].value_counts().reset_index(name='counts').query('counts > 1').values)
sorted_counts = list(dict(sorted(transaction_count.items())).values())


def aggregation(data=data_out, ids=block_ids):

    # define lists
    rel_counts, maximal_values, sum_of_values, mean_values = [], [], [], []

    for id in block_ids:

        # get only one block
        one_block = data_out[data_out['block_id'] == id]

        # filter for transactions bigger than 100 BTCs and get relative counts
        big_tx_one_block = one_block[one_block['value'] > 10**10]
        number_big_txs, number_all_txs = len(big_tx_one_block), len(one_block)
        rel_counts_big_all = number_big_txs/number_all_txs
        rel_counts.append(rel_counts_big_all)

        # get maximal value in a block
        maximal_values.append(max(one_block['value']))

        # get sum of all values in a block
        sum_of_values.append(sum(one_block['value']))

        # get mean of all values in a block
        mean_values.append(np.mean(one_block['value']))

    return maximal_values, mean_values, sum_of_values, rel_counts


# create the dataset
data = {
    'block_id': block_ids,
    'time': time_of_a_block,
    'number_of_txs': sorted_counts,
    'maximal_satoshi': aggregation(data_out, block_ids)[0],
    'mean_of_satoshi': aggregation(data_out, block_ids)[1],
    'sum_of_satoshi': aggregation(data_out, block_ids)[2],
    'relative_counts_>_100_B': aggregation(data_out, block_ids)[3],
    'max_fee_per_kb': fee_per_kb
}

# convert to dataframe structure
dataset = pd.DataFrame(data)

# .csv output (you can find it saved in the file of this project)
dataset.to_csv('btc_dataset', sep='\t')
