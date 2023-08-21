## How a groupby works?


### Original Table
Imagine we have a table of sales data with columns `Year`, `Quarter`, and `Revenue`.

| Year | Quarter | Revenue |
|------|---------|---------|
| 2020 | Q1      | 100     |
| 2020 | Q1      | 200     |
| 2020 | Q2      | 300     |
| 2021 | Q1      | 400     |
| 2021 | Q2      | 200     |
| 2021 | Q1      | 100     |
| 2021 | Q2      | 300     |

### Using `groupby` to Summarize by Year
We could use `groupby` to group the data by `Year` and sum the `Revenue`.

The resulting table might look something like this:

| Year | Total Revenue |
|------|---------------|
| 2020 | 600           |
| 2021 | 1000          |

### Using `groupby` to Summarize by Year and Quarter

We could go further and group the data by both `Year` and `Quarter`, again summing the `Revenue`.

| Year | Quarter | Total Revenue |
|------|---------|---------------|
| 2020 | Q1      | 300           |
| 2020 | Q2      | 300           |
| 2021 | Q1      | 500           |
| 2021 | Q2      | 500           |


## How hashing can be used to write the groupby algorithm?

### Original Table

| Year | Quarter | Revenue |
|------|---------|---------|
| 2020 | Q1      | 100     |
| 2020 | Q1      | 200     |
| 2020 | Q2      | 300     |
| 2021 | Q1      | 400     |
| 2021 | Q2      | 200     |
| 2021 | Q1      | 100     |
| 2021 | Q2      | 300     |

### Step 1: Initialize Hash Table

We start by initializing an empty hash table.

- Hash Table: `{}`

### Step 2: Hash the Keys and Sum the Values

We iterate through each row, hashing the `Year` and `Quarter` columns to form a key, and then adding the `Revenue` to the appropriate key-value pair in the hash table.

#### Row-wise Iteration:

- **Row 1:**
  - **Key = 2020-Q1**
  - **Value = 100**
  
  Hash Table becomes: `{'2020-Q1': 100}`

- **Row 2:**
  - **Key = 2020-Q1**
  - **Value = 200**
  
  Hash Table becomes: `{'2020-Q1': 300}` (existing key, so add value to existing sum)

- **Row 3:**
  - **Key = 2020-Q2**
  - **Value = 300**
  
  Hash Table becomes: `{'2020-Q1': 300, '2020-Q2': 300}`

- **Row 4:**
  - **Key = 2021-Q1**
  - **Value = 400**
  
  Hash Table becomes: `{'2020-Q1': 300, '2020-Q2': 300, '2021-Q1': 400}`

- **Row 5:**
  - **Key = 2021-Q2**
  - **Value = 200**
  
  Hash Table becomes: `{'2020-Q1': 300, '2020-Q2': 300, '2021-Q1': 400, '2021-Q2': 200}`

- **Row 6:**
  - **Key = 2021-Q1**
  - **Value = 100**
  
  Hash Table becomes: `{'2020-Q1': 300, '2020-Q2': 300, '2021-Q1': 500, '2021-Q2': 200}` (existing key, so add value to existing sum)

- **Row 7:**
  - **Key = 2021-Q2**
  - **Value = 300**
  
  Hash Table becomes: `{'2020-Q1': 300, '2020-Q2': 300, '2021-Q1': 500, '2021-Q2': 500}` (existing key, so add value to existing sum)

#### Final Hash Table Visualization

The hash table at this stage:

| Key      | Value |
|----------|-------|
| 2020-Q1  | 300   |
| 2020-Q2  | 300   |
| 2021-Q1  | 500   |
| 2021-Q2  | 500   |

### Step 3: Create the New Table from the Hash Table

Lastly, we'll convert this hash table into our grouped table.

| Year | Quarter | Total Revenue |
|------|---------|---------------|
| 2020 | Q1      | 300           |
| 2020 | Q2      | 300           |
| 2021 | Q1      | 500           |
| 2021 | Q2      | 500           |

By iterating through every row and using a hash table, we've efficiently implemented the `groupby` operation. The hash table allows us to quickly allocate each revenue value to the appropriate year and quarter, summing them up as we go.

---

## How a distributed groupby works? 

## Distributed `groupby` using Hash-based Partitioning

### 1. Data Loading

Let's assume we still have the same Sales Data table:

| Year | Quarter | Revenue |
|------|---------|---------|
| 2020 | Q1      | 100     |
| 2020 | Q1      | 200     |
| 2020 | Q2      | 300     |
| 2021 | Q1      | 400     |
| 2021 | Q2      | 200     |
| 2021 | Q1      | 100     |
| 2021 | Q2      | 300     |
| 2022 | Q1      | 150     |
| 2022 | Q2      | 250     |

### 2. Data Partitioning Stage

#### Step 1: Hash-based Partitioning

We'll use the combination of `Year` and `Quarter` as the hash key for partitioning the data across nodes. Here, let's assume the hash function puts similar keys (i.e., the same `Year` and `Quarter`) on the same node.

##### Node 1:

| Year | Quarter | Revenue |
|------|---------|---------|
| 2020 | Q1      | 100     |
| 2020 | Q1      | 200     |
| 2021 | Q1      | 400     |
| 2021 | Q1      | 100     |

##### Node 2:

| Year | Quarter | Revenue |
|------|---------|---------|
| 2020 | Q2      | 300     |
| 2021 | Q2      | 200     |
| 2021 | Q2      | 300     |

##### Node 3:

| Year | Quarter | Revenue |
|------|---------|---------|
| 2022 | Q1      | 150     |
| 2022 | Q2      | 250     |

#### Step 2: Local GroupBy on Each Node

Each node performs a local `groupby` operation on the data.

**Node 1 Local GroupBy:**

| Year | Quarter | Local Total Revenue |
|------|---------|---------------------|
| 2020 | Q1      | 300                 |
| 2021 | Q1      | 500                 |

**Node 2 Local GroupBy:**

| Year | Quarter | Local Total Revenue |
|------|---------|---------------------|
| 2020 | Q2      | 300                 |
| 2021 | Q2      | 500                 |

**Node 3 Local GroupBy:**

| Year | Quarter | Local Total Revenue |
|------|---------|---------------------|
| 2022 | Q1      | 150                 |
| 2022 | Q2      | 250                 |

### 3. Merge Stage

#### Step 3: Merge Partial GroupBy Results

In a hash-based distributed setting, all identical keys should be located on the same node after partitioning. Therefore, the local `groupby` on each node should actually give us the final `groupby` result, removing the need for an additional merge step across nodes for the same keys.

**Final Result:**

| Year | Quarter | Final Total Revenue |
|------|---------|---------------------|
| 2020 | Q1      | 300                 |
| 2020 | Q2      | 300                 |
| 2021 | Q1      | 500                 |
| 2021 | Q2      | 500                 |
| 2022 | Q1      | 150                 |
| 2022 | Q2      | 250                 |

The keys that ended up on the same node were correctly grouped together locally, providing us with the correct global `groupby` result.