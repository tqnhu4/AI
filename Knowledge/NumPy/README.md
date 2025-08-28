
-----

## NumPy Cheat Sheet

NumPy (Numerical Python) is the fundamental package for numerical computation in Python. It provides an array object of arbitrary homogeneous items, and many routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation, etc.

-----

### 1\. Installation

```bash
pip install numpy
```

-----

### 2\. Importing NumPy

```python
import numpy as np
```

-----

### 3\. Array Creation

| Method | Description | Example | Result |
| :------------------------------ | :---------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------- | :----------------------------------- |
| `np.array()` | Create an array from a Python list or tuple. | `np.array([1, 2, 3])` \<br\> `np.array([[1, 2], [3, 4]])` | `[1 2 3]` \<br\> `[[1 2]` \<br\> `  [3 4]] ` |
| `np.zeros()` | Create an array filled with zeros. | `np.zeros(3)` \<br\> `np.zeros((2, 3))` | `[0. 0. 0.]` \<br\> `[[0. 0. 0.]` \<br\> `  [0. 0. 0.]] ` |
| `np.ones()` | Create an array filled with ones. | `np.ones(3)` \<br\> `np.ones((2, 3))` | `[1. 1. 1.]` \<br\> `[[1. 1. 1.]` \<br\> `  [1. 1. 1.]] ` |
| `np.empty()` | Create an array without initializing entries (may contain garbage). | `np.empty(3)` | `[x. x. x.]` (values depend on memory) |
| `np.arange()` | Create an array with a range of values. | `np.arange(5)` \<br\> `np.arange(1, 10, 2)` | `[0 1 2 3 4]` \<br\> `[1 3 5 7 9]` |
| `np.linspace()` | Create an array with evenly spaced values over a specified interval. | `np.linspace(0, 1, 5)` | `[0.   0.25 0.5  0.75 1.  ]` |
| `np.full()` | Create an array of a specified shape filled with a specified value. | `np.full((2, 2), 7)` | `[[7 7]` \<br\> `  [7 7]] ` |
| `np.eye()` | Create an identity matrix. | `np.eye(3)` | `[[1. 0. 0.]` \<br\> `  [0. 1. 0.] ` \<br\> `  [0. 0. 1.]] ` |
| `np.random.rand()` | Random floats in [0, 1) | `np.random.rand(2,2)` | `[[0.34 0.98]` \<br\> `  [0.12 0.76]] ` (example) |
| `np.random.randn()` | Random floats from standard normal dist. | `np.random.randn(2,2)` | `[[-0.85  0.37]` \<br\> `  [ 1.29 -0.01]] ` (example) |
| `np.random.randint()` | Random integers | `np.random.randint(0, 10, size=(2,2))` | `[[5 9]` \<br\> `  [1 3]] ` (example) |

-----

### 4\. Array Attributes

| Attribute | Description | Example (for `a = np.array([[1,2],[3,4],[5,6]])`) | Result |
| :-------------------- | :------------------------------------------------- | :------------------------------------------------ | :-------------------------------- |
| `.ndim` | Number of dimensions. | `a.ndim` | `2` |
| `.shape` | Tuple of array dimensions. | `a.shape` | `(3, 2)` |
| `.size` | Total number of elements. | `a.size` | `6` |
| `.dtype` | Data type of the elements. | `a.dtype` | `dtype('int32')` or `dtype('int64')` |
| `.itemsize` | Size of each element in bytes. | `a.itemsize` | `4` (for int32) or `8` (for int64) |
| `.nbytes` | Total bytes consumed by the array elements. | `a.nbytes` | `24` (for 6 int32 elements) |

-----

### 5\. Array Indexing and Slicing

```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 1D Array Indexing
arr[0]      # 0
arr[2:5]    # [2 3 4]
arr[:5]     # [0 1 2 3 4]
arr[5:]     # [5 6 7 8 9]
arr[::2]    # [0 2 4 6 8] (step of 2)
arr[::-1]   # [9 8 7 6 5 4 3 2 1 0] (reversed)

# 2D Array Indexing (matrix[row, col])
matrix[0, 0]    # 1
matrix[1, 2]    # 6
matrix[:, 1]    # [2 5 8] (all rows, 2nd column)
matrix[0, :]    # [1 2 3] (1st row, all columns)
matrix[0:2, 1:3] # [[2 3]
                 #  [5 6]]
```

-----

### 6\. Array Manipulation

| Method | Description | Example |
| :---------------------- | :---------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------- |
| `.reshape()` | Gives a new shape to an array without changing its data. | `arr.reshape(2, 5)` |
| `.resize()` | Modifies the array in place with a new shape. | `arr.resize(2, 5)` |
| `.flatten()` | Returns a copy of the array collapsed into one dimension. | `matrix.flatten()` |
| `.ravel()` | Returns a contiguous flattened array. | `matrix.ravel()` |
| `np.concatenate()` | Join a sequence of arrays along an existing axis. | `np.concatenate((a1, a2), axis=0)` |
| `np.vstack()` | Stack arrays in sequence vertically (row wise). | `np.vstack((a1, a2))` |
| `np.hstack()` | Stack arrays in sequence horizontally (column wise). | `np.hstack((a1, a2))` |
| `np.split()` | Split an array into multiple sub-arrays. | `np.split(arr, 2)` |
| `np.hsplit()` | Split an array into multiple sub-arrays horizontally. | `np.hsplit(matrix, 3)` |
| `np.vsplit()` | Split an array into multiple sub-arrays vertically. | `np.vsplit(matrix, 3)` |
| `.transpose()` or `.T` | Permute the dimensions of an array. | `matrix.transpose()` or `matrix.T` |
| `.newaxis` | Used to increase the dimension of the existing array by one more dimension. | `arr[:, np.newaxis]` or `arr[np.newaxis, :]` |

-----

### 7\. Basic Operations

NumPy supports element-wise operations.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise addition
a + b       # [5 7 9]
np.add(a, b) # [5 7 9]

# Element-wise subtraction
a - b       # [-3 -3 -3]
np.subtract(a, b) # [-3 -3 -3]

# Element-wise multiplication
a * b       # [4 10 18]
np.multiply(a, b) # [4 10 18]

# Element-wise division
b / a       # [4.   2.5  2.  ]
np.divide(b, a) # [4.   2.5  2.  ]

# Scalar operations
a + 10      # [11 12 13]
a * 2       # [2 4 6]
```

-----

### 8\. Aggregate Functions

| Function | Description | Example (for `arr = np.array([1, 2, 3, 4, 5, 6])`) | Result |
| :-------------------- | :------------------------------------------------------- | :--------------------------------------------- | :------- |
| `np.sum()` | Sum of array elements over a given axis. | `np.sum(arr)` \<br\> `matrix.sum(axis=0)` (column sum) | `21` \<br\> `[12 15 18]` |
| `np.min()` / `.min()` | Minimum of array elements. | `np.min(arr)` \<br\> `matrix.min(axis=1)` (row min) | `1` \<br\> `[1 4 7]` |
| `np.max()` / `.max()` | Maximum of array elements. | `np.max(arr)` \<br\> `matrix.max(axis=0)` | `6` \<br\> `[7 8 9]` |
| `np.mean()` / `.mean()` | Arithmetic mean. | `np.mean(arr)` \<br\> `matrix.mean()` (overall mean) | `3.5` \<br\> `5.0` |
| `np.std()` / `.std()` | Standard deviation. | `np.std(arr)` | `1.7078...` |
| `np.var()` / `.var()` | Variance. | `np.var(arr)` | `2.9166...` |
| `np.median()` | Median. | `np.median(arr)` | `3.5` |
| `np.cumsum()` | Cumulative sum of elements. | `np.cumsum(arr)` | `[1  3  6 10 15 21]` |
| `np.cumprod()` | Cumulative product of elements. | `np.cumprod(arr)` | `[1   2   6  24 120 720]` |

-----

### 9\. Linear Algebra

| Function | Description | Example |
| :---------------------- | :-------------------------------------- | :---------------------------------------- |
| `np.dot()` | Dot product of two arrays. | `np.dot(a, b)` \<br\> `np.dot(matrix1, matrix2)` |
| `np.linalg.det()` | Compute the determinant of an array. | `np.linalg.det(matrix)` |
| `np.linalg.inv()` | Compute the (multiplicative) inverse of a matrix. | `np.linalg.inv(matrix)` |
| `np.linalg.eig()` | Compute the eigenvalues and right eigenvectors of a square array. | `np.linalg.eig(matrix)` |
| `np.linalg.solve()` | Solve a linear matrix equation, or system of linear scalar equations. | `np.linalg.solve(A, b)` |

-----

### 10\. Broadcasting

NumPy's broadcasting allows universal functions to operate on arrays of different shapes.

```python
a = np.array([[1, 2, 3], [4, 5, 6]]) # shape (2, 3)
b = np.array([10, 20, 30])         # shape (3,)

result = a + b
# Result:
# [[11 22 33]
#  [14 25 36]]

c = np.array([[10], [20]])         # shape (2, 1)
result2 = a + c
# Result:
# [[11 12 13]
#  [24 25 26]]
```

-----

### 11\. Copying Arrays

```python
a = np.array([1, 2, 3])
b = a          # b is just a reference/view of a (changes to b affect a)
c = a.copy()   # c is a new array (changes to c do NOT affect a)
```

-----

### 12\. Saving and Loading Arrays

```python
data = np.arange(10)

# Save to a .npy file (NumPy's binary format)
np.save('my_array.npy', data)

# Load from a .npy file
loaded_data = np.load('my_array.npy')

# Save to a text file (e.g., CSV)
np.savetxt('my_array.csv', data, delimiter=',')

# Load from a text file
loaded_txt_data = np.loadtxt('my_array.csv', delimiter=',')
```

-----

This cheat sheet covers the most commonly used functionalities of NumPy. For more in-depth information, always refer to the [official NumPy documentation](https://numpy.org/doc/stable/).