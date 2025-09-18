# Statistical-Foundation-of-Data-Sciences

## How to run

1. Open Google Colab or a new Jupyter notebook.
2. Create a new code cell and paste the code blocks given below in sequence.
3. Execute cells top-to-bottom.

---

## Dependencies

```python
import numpy as np
import pandas as pd
```

---

## Dataset creation (seed = 42)

We'll create a DataFrame with the following columns: `age`, `income`. We'll intentionally insert `NaN`s in `income` and `age`.

```python
np.random.seed(42)

number_records = 150

data = {
    "age": np.random.randint(18, 50, size=number_records),
    "income": np.random.randint(20000, 50000, size=number_records),
}

df = pd.DataFrame(data)

nan_indices_income = np.random.choice(df.index, 50, replace=False)
nan_indices_score = np.random.choice(df.index, 30, replace=False)

df.loc[nan_indices_income, "income"] = np.nan

df

---
```

# Problem 1 — mean, median, age-weighted mean of income

Requirements:
- Ignore `NaN`s appropriately (i.e., do not let `NaN` propagate when computing these summary stats).
- Explain when weighted mean is preferable.

```python
# mean of income
mean_income = df['income'].mean()
print(f"Mean Income: {mean_income:,.2f}")

# median of incom
median_income = df['income'].median()
print(f"Median Income: {median_income:,.2f}")

# age-weighted mean of income
temp_df = df.dropna(subset=['income', 'age'])
age_weighted_mean_income = np.average(temp_df['income'], weights=temp_df['age'])
print(f"Age-Weighted Mean Income: {age_weighted_mean_income:,.2f}")
```

---

# Problem 2 — Standardize income (z-score) and count outliers

Requirements:
- Compute z-score ignoring `NaN`s.
- Count incomes with |z| > 3 as outliers.
- Do not drop entire rows unnecessarily.

```python
# Compute z-score manually to control NaN handling
income_mean = df['income'].mean(skipna=True)
income_std = df['income'].std(skipna=True)

# Create a z-score column that stays NaN wherever income is NaN
income_mean = df['income'].mean()
income_std = df['income'].std()

# z score
df['income_zscore'] = (df['income'] - income_mean) / income_std

# z>3 (outliers)
num_outliers = (df['income_zscore'].abs() > 3).sum()

print(f"Mean Income for Z-Score Calculation: {income_mean:.2f}")
print(f"Std Dev of Income for Z-Score Calculation: {income_std:.2f}\n")
print("First 5 rows with Z-scores:")
print(df[['income', 'income_zscore']].head())
print(f"\nNumber of income outliers (|z| > 3): {num_outliers}")
```

**Notes about NaN handling:**
- We computed mean and std using `skipna=True`, so `NaN` incomes do not affect the population mean/std.
- We kept rows with NaN incomes present in the DataFrame; only the `income_z` entry for those rows is `NaN`.

---

# Problem 3 — Age bins and statistics

Requirements:
- Age bins: `[18-25), [25-35), [35-45), [45-60)`
- For each bin compute: count, mean income, median score.
- Show as a tidy DataFrame sorted by age bin.

```python
bins = [18, 25, 35, 45, 60]

df['age_bin'] = pd.cut(df['age'], bins=bins, right=False)

age_bin_analysis = (
    df.groupby('age_bin', observed=True)
    .agg(count_of_observations=('age', 'count'), mean_income=('income', 'mean')))

age_bin_analysis
```

**Explanation:**
- `count_obs` counts how many rows fall into each age bin; it does not drop rows where `income` or `score` is `NaN`.
- `mean_income` and `median_score` compute statistics skipping `NaN`s in those respective columns.

---

# Problem 4 — Create an array (not 1D) and showcase operations

Requirements:
- Create an array that is not 1D (e.g., 2D or 3D).
- Demonstrate: shape, size, transpose, flatten.
- Demonstrate negative indexing and display an error while doing invalid slicing.
- Arithmetic operations: broadcasting and dot product.
- Linear algebra: determinant and inverse.

```python
# 2D array
my_array = np.arange(12).reshape(3, 4)
print("Original Array:")
print(my_array)

# Shape and Resize
print("\nShape and Resize Operations")
print(f"Shape of the array: {my_array.shape}")
print(f"Size (total elements) of the array: {my_array.size}")
print("Transposed Array (T):")
print(my_array.T)
print("Flattened Array (1D):")
print(my_array.flatten())a

# Negative Indexing and Slicing Error
print("\nIndexing and Slicing")
# Negative indexing gets the last element from the last row
last_element = my_array[-1, -1]
print(f"Last element using negative indexing my_array[-1, -1]: {last_element}")

# Displaying an error while slicing
try:
    error_slice = my_array[5, 5]
except IndexError as e:
    print(f"\nSuccessfully caught an expected error: {e}")


# Arithmetic Operations
print("\nArithmetic Operations")
# Broadcasting: adding a scalar to the array
broadcasted_array = my_array + 100
print("Broadcasting (Array + 100):")
print(broadcasted_array)

# Dot Product
array_b = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
print("\nSecond array for dot product (shape 4x2):")
print(array_b)

dot_product = my_array @ array_b # Using the @ operator
print("Dot Product of a 3x4 and 4x2 array (result is 3x2):")
print(dot_product)

# Linear Algebra
print("\nLinear Algebra")
# Determinant and Inverse require a square matrix (NxN)
square_matrix = np.array([[4, 7], [2, 6]])
print("Square Matrix:")
print(square_matrix)

# Calculate the determinant
determinant = np.linalg.det(square_matrix)
print(f"\nDeterminant of the square matrix: {determinant:.2f}")

# Calculate the inverse
if determinant != 0:
    inverse_matrix = np.linalg.inv(square_matrix)
    print("Inverse of the square matrix:")
    print(inverse_matrix)
else:
    print("Matrix does not have an inverse (determinant is zero).")
```

---

