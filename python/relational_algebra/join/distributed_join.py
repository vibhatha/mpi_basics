import pandas as pd
from mpi4py import MPI

def hash_partition(table, num_partitions, key_index):
    """Partition table based on hash of join key."""
    partitions = [[] for _ in range(num_partitions)]
    for row in table:
        partition = hash(row[key_index]) % num_partitions
        partitions[partition].append(row)
    return partitions

def local_hash_join(table1, table2, table1_key_index, table2_key_index):
    hash_table = {}
    for record in table1:
        key = record[table1_key_index]
        if key not in hash_table:
            hash_table[key] = []
        hash_table[key].append(record)

    results = []
    for record in table2:
        key = record[table2_key_index]
        if key in hash_table:
            for match in hash_table[key]:
                combined_record = match + record[:table2_key_index] + record[table2_key_index+1:]
                results.append(combined_record)
    return results

def distributed_hash_join(df1, df2, key1, key2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Convert DataFrames to lists of tuples
    table1 = [tuple(row) for _, row in df1.iterrows()]
    table2 = [tuple(row) for _, row in df2.iterrows()]

    table1_key_index = df1.columns.get_loc(key1)
    table2_key_index = df2.columns.get_loc(key2)

    # 1. Partitioning Phase
    partitions1 = hash_partition(table1, size, table1_key_index)
    partitions2 = hash_partition(table2, size, table2_key_index)

    local_table1 = comm.scatter(partitions1, root=0)
    local_table2 = comm.scatter(partitions2, root=0)

    # 2. Join Phase
    local_results = local_hash_join(local_table1, local_table2, table1_key_index, table2_key_index)

    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        flat_results = [item for sublist in all_results for item in sublist]
        columns = list(df1.columns) + [col for col in df2.columns if col != key2]
        joined_df = pd.DataFrame(flat_results, columns=columns)
        return joined_df
    return None

# Sample data
df1 = pd.DataFrame({
    'key': [1, 2, 3, 5],
    'value1': ['a', 'b', 'c', 'e']
})

df2 = pd.DataFrame({
    'key': [2, 3, 4, 5],
    'value2': ['x', 'y', 'z', 'w']
})

# Use the custom distributed hash join
joined_distributed = distributed_hash_join(df1, df2, 'key', 'key')

# Use pandas built-in merge for comparison
joined_pandas = df1.merge(df2, left_on='key', right_on='key', how='inner')

if MPI.COMM_WORLD.Get_rank() == 0:
    print("Distributed Join Results:")
    print(joined_distributed)
    print("\nPandas Join Results:")
    print(joined_pandas)
