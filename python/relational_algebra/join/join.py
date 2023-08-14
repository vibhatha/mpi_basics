import pandas as pd


def hash_join(df1, df2, key1, key2):
    # Convert DataFrames to list of tuples
    table1 = [tuple(row) for _, row in df1.iterrows()]
    table2 = [tuple(row) for _, row in df2.iterrows()]

    # Indexes for join keys
    table1_key_index = df1.columns.get_loc(key1)
    table2_key_index = df2.columns.get_loc(key2)

    # Size Decision
    if len(table1) <= len(table2):
        build_table, probe_table, build_key_index, probe_key_index = table1, table2, table1_key_index, table2_key_index
    else:
        build_table, probe_table, build_key_index, probe_key_index = table2, table1, table2_key_index, table1_key_index

    # Build Stage
    hash_table = {}
    for record in build_table:
        key = record[build_key_index]
        if key not in hash_table:
            hash_table[key] = []
        hash_table[key].append(record)

    # Probe Stage
    results = []
    for record in probe_table:
        key = record[probe_key_index]
        if key in hash_table:
            for match in hash_table[key]:
                # Remove the key from the second table's record when appending
                combined_record = match + record[:probe_key_index] + record[probe_key_index+1:]
                results.append(combined_record)

    # Convert results to DataFrame
    columns = list(df1.columns) + [col for col in df2.columns if col != key2]
    joined_df = pd.DataFrame(results, columns=columns)
    return joined_df


# Sample data
df1 = pd.DataFrame({
    'key': [1, 2, 3],
    'value1': ['a', 'b', 'c']
})

df2 = pd.DataFrame({
    'key': [2, 3, 4],
    'value2': ['x', 'y', 'z']
})

# Using our custom hash join function
joined_custom = hash_join(df1, df2, 'key', 'key')

# Using pandas merge function for comparison
joined_pandas = df1.merge(df2, left_on='key', right_on='key', how='inner')

# Compare the results
print("Custom Join")
print(joined_custom)
print("Pandas Join")
print(joined_pandas)
