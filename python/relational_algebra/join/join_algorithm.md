# Join Algorithm


## How a hash join works?

Let's say we have two tables `Employees` and `Departments`.

**Employees:**

| EmployeeID | Name      | DepartmentID |
|------------|-----------|--------------|
| 1          | Alice     | 100          |
| 2          | Bob       | 200          |
| 3          | Carol     | 300          |
| 4          | Dave      | 100          |
| 5          | Eve       | 400          |

**Departments:**

| DepartmentID | DepartmentName  |
|--------------|-----------------|
| 100          | HR              |
| 200          | Sales           |
| 300          | Engineering     |
| 400          | Marketing       |

We want to join these tables based on `DepartmentID`.

## Steps

1. **Hashing Phase:** Build a hash table from one of the tables, often the smaller one. Here, let's hash the `Departments` table. The hash table might look something like this, using `DepartmentID` as the hash key:

    ```
    Hash Table
    ---------------------------
    Key: 100  Value: HR
    Key: 200  Value: Sales
    Key: 300  Value: Engineering
    Key: 400  Value: Marketing
    ```

2. **Probe Phase:** Go through each row in the other table (`Employees` in this case) and look for matching rows in the hash table.

## Visualizing the Process

1. **First Row in Employees:** Alice with `DepartmentID = 100`
    - Hash `100` and look it up in the hash table.
    - Find that `100` corresponds to `HR`.
    - Add a joined row: `[1, Alice, 100, HR]`

2. **Second Row in Employees:** Bob with `DepartmentID = 200`
    - Hash `200` and look it up in the hash table.
    - Find that `200` corresponds to `Sales`.
    - Add a joined row: `[2, Bob, 200, Sales]`

3. **Third Row in Employees:** Carol with `DepartmentID = 300`
    - Hash `300` and look it up in the hash table.
    - Find that `300` corresponds to `Engineering`.
    - Add a joined row: `[3, Carol, 300, Engineering]`

4. **Fourth Row in Employees:** Dave with `DepartmentID = 100`
    - Hash `100` and look it up in the hash table.
    - Find that `100` corresponds to `HR`.
    - Add a joined row: `[4, Dave, 100, HR]`

5. **Fifth Row in Employees:** Eve with `DepartmentID = 400`
    - Hash `400` and look it up in the hash table.
    - Find that `400` corresponds to `Marketing`.
    - Add a joined row: `[5, Eve, 400, Marketing]`

## Resulting Joined Table

| EmployeeID | Name  | DepartmentID | DepartmentName |
|------------|-------|--------------|----------------|
| 1          | Alice | 100          | HR             |
| 2          | Bob   | 200          | Sales          |
| 3          | Carol | 300          | Engineering    |
| 4          | Dave  | 100          | HR             |
| 5          | Eve   | 400          | Marketing      |

Hash joins are very efficient for large datasets because they minimize disk I/O and take advantage of the high speed of in-memory hash table lookups. However, they do require enough memory to store the hash table, which can be a limitation.

## How a distributed hash join works?


### Node A:

**Employees:**

| EmployeeID | Name      | DepartmentID |
|------------|-----------|--------------|
| 1          | Alice     | 100          |
| 2          | Bob       | 200          |
| 6          | Frank     | 200          |

**Departments:**

| DepartmentID | DepartmentName  |
|--------------|-----------------|
| 100          | HR              |
| 200          | Sales           |

### Node B:

**Employees:**

| EmployeeID | Name      | DepartmentID |
|------------|-----------|--------------|
| 3          | Carol     | 300          |
| 4          | Dave      | 100          |
| 7          | Grace     | 400          |

**Departments:**

| DepartmentID | DepartmentName  |
|--------------|-----------------|
| 300          | Engineering     |
| 400          | Marketing       |

### Node C:

**Employees:**

| EmployeeID | Name      | DepartmentID |
|------------|-----------|--------------|
| 5          | Eve       | 500          |
| 8          | Helen     | 300          |
| 9          | Ian       | 500          |

**Departments:**

| DepartmentID | DepartmentName  |
|--------------|-----------------|
| 500          | Finance         |

### Steps for Distributed Hash Join:

1. **Local Hashing Phase:** Each node builds a local hash table from its portion of the `Departments` table.

    - **Node A Hash Table:**
        ```
        Key: 100  Value: HR
        Key: 200  Value: Sales
        ```
    - **Node B Hash Table:**
        ```
        Key: 300  Value: Engineering
        Key: 400  Value: Marketing
        ```
    - **Node C Hash Table:**
        ```
        Key: 500  Value: Finance
        ```

2. **Data Redistribution:** Each node partitions its `Employees` data based on the same hash function and sends it to the corresponding node.

    - Alice, Dave (`DepartmentID = 100`) hash to Node A.
    - Bob, Frank (`DepartmentID = 200`) stay in Node A.
    - Carol, Helen (`DepartmentID = 300`) hash to Node B.
    - Grace (`DepartmentID = 400`) stays in Node B.
    - Eve, Ian (`DepartmentID = 500`) hash to Node C.

3. **Local Join Phase:** Each node performs a hash join locally.

    - **Node A Joined Data:**
        ```
        [1, Alice, 100, HR]
        [4, Dave, 100, HR]
        [2, Bob, 200, Sales]
        [6, Frank, 200, Sales]
        ```
    - **Node B Joined Data:**
        ```
        [3, Carol, 300, Engineering]
        [8, Helen, 300, Engineering]
        [7, Grace, 400, Marketing]
        ```
    - **Node C Joined Data:**
        ```
        [5, Eve, 500, Finance]
        [9, Ian, 500, Finance]
        ```

4. **Combine Results:** The joined data from all nodes are combined.

    | EmployeeID | Name  | DepartmentID | DepartmentName |
    |------------|-------|--------------|----------------|
    | 1          | Alice | 100          | HR             |
    | 4          | Dave  | 100          | HR             |
    | 2          | Bob   | 200          | Sales          |
    | 6          | Frank | 200          | Sales          |
    | 3          | Carol | 300          | Engineering    |
    | 8          | Helen | 300          | Engineering    |
    | 7          | Grace | 400          | Marketing      |
    | 5          | Eve   | 500          | Finance        |
    | 9          | Ian   | 500          | Finance        |

With more data and more nodes, the process remains largely the same, just more distributed.

## How a sort-merge join works?

A Sort-Merge Join is a join algorithm used to combine rows from two or more tables based on a related column between them. The algorithm is generally used in the context of database systems to optimize query performance. The basic steps in the algorithm are:

    Sort: Sort each table on the join key.
    Merge: Traverse each sorted list once, merging matching rows.

Certainly! When there are multiple rows with the same ID in both tables, a sort-merge join creates all possible combinations of rows that share the same ID from both tables. This is often referred to as a "cross product" for each common ID value.

Let's consider larger tables with duplicate IDs for a more in-depth illustration.

### Tables Before Sorting

**Table A:**

| ID | Name    |
|----|---------|
| 1  | Alice   |
| 3  | Carol   |
| 2  | Bob     |
| 1  | Amy     |
| 3  | Calvin  |

**Table B:**

| ID | Score   |
|----|---------|
| 2  | 90      |
| 1  | 85      |
| 4  | 95      |
| 3  | 92      |
| 1  | 88      |

---

### Step 1: Sort Both Tables on ID

**Sorted Table A:**

| ID | Name    |
|----|---------|
| 1  | Alice   |
| 1  | Amy     |
| 2  | Bob     |
| 3  | Carol   |
| 3  | Calvin  |

**Sorted Table B:**

| ID | Score   |
|----|---------|
| 1  | 85      |
| 1  | 88      |
| 2  | 90      |
| 3  | 92      |
| 4  | 95      |

---

### Step 2: Initialize Pointers

We initialize pointers at the beginning of each sorted table.

**Pointer A ->**  
| ID | Name    |
|----|---------|
| 1  | Alice   |
| 1  | Amy     |
| 2  | Bob     |
| 3  | Carol   |
| 3  | Calvin  |

**Pointer B ->**  
| ID | Score   |
|----|---------|
| 1  | 85      |
| 1  | 88      |
| 2  | 90      |
| 3  | 92      |
| 4  | 95      |

**Result Table:**  
(empty)

---

### Step 3: Compare and Merge

#### Iteration 1:

- **Compare IDs**: Both pointers are at `ID = 1`.
- **Action**: Merge all combinations and move both pointers.

**Result Table:**  
| ID | Name  | Score |
|----|-------|-------|
| 1  | Alice | 85    |
| 1  | Alice | 88    |
| 1  | Amy   | 85    |
| 1  | Amy   | 88    |

---

#### Iteration 2:

- **Compare IDs**: Both pointers are at `ID = 2`.
- **Action**: Merge and move both pointers.

**Result Table:**  
| ID | Name  | Score |
|----|-------|-------|
| 1  | Alice | 85    |
| 1  | Alice | 88    |
| 1  | Amy   | 85    |
| 1  | Amy   | 88    |
| 2  | Bob   | 90    |

---

#### Iteration 3:

- **Compare IDs**: Both pointers are at `ID = 3`.
- **Action**: Merge all combinations and move both pointers.

**Result Table:**  
| ID | Name  | Score |
|----|-------|-------|
| 1  | Alice | 85    |
| 1  | Alice | 88    |
| 1  | Amy   | 85    |
| 1  | Amy   | 88    |
| 2  | Bob   | 90    |
| 3  | Carol | 92    |
| 3  | Calvin| 92    |

---

#### Iteration 4:

- **Compare IDs**: Pointer A is at the end, Pointer B is at `ID = 4`.
- **Action**: Terminate.

---

The Result Table contains all combinations of rows from `Table A` and `Table B` that have the same ID. This is particularly useful when dealing with "one-to-many" or "many-to-many" relationships between tables.

---

## How a distributed sort-merge join works?

Certainly! Here is a more comprehensive explanation, including the partitioning of both the Employees and Departments tables across the nodes.

---

## How a Distributed Hash Join Works with Data Partitioning

### Original Data on Each Node

#### Node A

**Employees:**

| EmployeeID | Name  | DepartmentID |
|------------|-------|--------------|
| 1          | Alice | 100          |
| 2          | Bob   | 200          |
| 6          | Frank | 200          |

**Departments:**

| DepartmentID | DepartmentName |
|--------------|----------------|
| 100          | HR             |
| 200          | Sales          |

#### Node B

**Employees:**

| EmployeeID | Name  | DepartmentID |
|------------|-------|--------------|
| 3          | Carol | 300          |
| 4          | Dave  | 100          |
| 7          | Grace | 400          |

**Departments:**

| DepartmentID | DepartmentName |
|--------------|----------------|
| 300          | Engineering    |
| 400          | Marketing      |

#### Node C

**Employees:**

| EmployeeID | Name  | DepartmentID |
|------------|-------|--------------|
| 5          | Eve   | 500          |
| 8          | Helen | 300          |
| 9          | Ian   | 500          |

**Departments:**

| DepartmentID | DepartmentName |
|--------------|----------------|
| 500          | Finance        |

### Step 1: Data Partitioning

For partitioning, we use the hash function `DepartmentID % 3`.

#### Departments Table Partitioning

- **Node A partitions Departments:**
  - 100 % 3 = 1 → Send to Node B
  - 200 % 3 = 2 → Send to Node C
  
- **Node B partitions Departments:**
  - 300 % 3 = 0 → Send to Node A
  - 400 % 3 = 1 → Stays in Node B

- **Node C partitions Departments:**
  - 500 % 3 = 2 → Stays in Node C

#### Employees Table Partitioning

- **Node A partitions Employees:**
  - Alice, Frank (`DepartmentID = 100, 200`) → Send to Node B, Node C respectively
  - Bob (`DepartmentID = 200`) → Send to Node C
  
- **Node B partitions Employees:**
  - Carol (`DepartmentID = 300`) → Send to Node A
  - Dave, Grace (`DepartmentID = 100, 400`) → Send to Node B, Stay in Node B
  
- **Node C partitions Employees:**
  - Eve, Ian (`DepartmentID = 500`) → Stay in Node C
  - Helen (`DepartmentID = 300`) → Send to Node A

#### After Data Partitioning: Distributed Tables

**Node A:**

| DepartmentID | DepartmentName |
|--------------|----------------|
| 300          | Engineering    |

| EmployeeID | Name  | DepartmentID |
|------------|-------|--------------|
| 3          | Carol | 300          |
| 8          | Helen | 300          |

**Node B:**

| DepartmentID | DepartmentName |
|--------------|----------------|
| 100          | HR             |
| 400          | Marketing      |

| EmployeeID | Name  | DepartmentID |
|------------|-------|--------------|
| 1          | Alice | 100          |
| 4          | Dave  | 100          |
| 7          | Grace | 400          |

**Node C:**

| DepartmentID | DepartmentName |
|--------------|----------------|
| 200          | Sales          |
| 500          | Finance        |

| EmployeeID | Name  | DepartmentID |
|------------|-------|--------------|
| 2          | Bob   | 200          |
| 6          | Frank | 200          |
| 5          | Eve   | 500          |
| 9          | Ian   | 500          |

### Step 2: Local Hashing Phase

- **Node A Hash Table:**
    ```
    Key: 300  Value: Engineering
    ```
  
- **Node B Hash Table:**
    ```
    Key: 100  Value: HR
    Key: 400  Value: Marketing
    ```
  
- **Node C Hash Table:**
    ```
    Key: 200  Value: Sales
    Key: 500  Value: Finance
    ```

### Step 3: Local Join Phase

- **Node A Joined Data:**
    ```
    [3, Carol, 300, Engineering]
    [8, Helen, 300, Engineering]
    ```
  
- **Node B Joined Data:**
    ```
    [1, Alice, 100, HR]
    [4, Dave, 100, HR]
    [7, Grace, 400, Marketing]
    ```

- **Node C Joined Data:**
    ```
    [2, Bob, 200, Sales]
    [6, Frank, 200, Sales]
    [5, Eve, 500, Finance]
    [9, Ian, 500, Finance]
    ```

### Step 4: Combine Results

| EmployeeID | Name  | DepartmentID | DepartmentName |
|------------|-------|--------------|----------------|
| 1          | Alice | 100          | HR             |
| 4          | Dave  | 100          | HR             |
| 2          | Bob   | 200          | Sales          |
| 6          | Frank | 200          | Sales          |
| 3          | Carol | 300          | Engineering    |
| 8          | Helen | 300          | Engineering    |
| 7          | Grace | 400          | Marketing      |
| 5          | Eve   | 500          | Finance        |
| 9          | Ian   | 500          | Finance        |

---


## References

1. [Sort-Merge Join](https://en.wikipedia.org/wiki/Sort-merge_join)
2. [Hash Join](https://en.wikipedia.org/wiki/Hash_join)

