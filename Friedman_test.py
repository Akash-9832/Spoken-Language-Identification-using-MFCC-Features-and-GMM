import numpy as np
from scipy.stats import friedmanchisquare

# Dataset 1
# Accuracy data for each language and split ratio
# Rows: languages, Columns: split ratios (80:20, 70:30, 60:40, 50:50)
# accuracy_data = np.array([
#     [0.99, 0.98, 0.99, 0.99],  # Language 1
#     [0.99, 0.98, 0.98, 0.99],  # Language 2
#     [0.98, 0.98, 0.97, 0.98],  # Language 3
#     [1.00, 1.00, 1.00, 1.00],  # Language 4
#     [0.97, 0.94, 0.96, 0.94],  # Language 5
#     [1.00, 1.00, 1.00, 1.00],  # Language 6
#     [0.99, 0.99, 0.98, 0.98],  # Language 7
#     [0.97, 0.94, 0.95, 0.92]   # Language 8
# ])

# # Apply the Friedman test
# statistic, p_value = friedmanchisquare(accuracy_data[:, 0], accuracy_data[:, 1], accuracy_data[:, 2], accuracy_data[:, 3])

# print(f'Friedman Test Statistic: {statistic}')
# print(f'P-value: {p_value}')

# # Interpret the result
# if p_value < 0.05:
#     print('The differences between the split ratios are statistically significant.')
# else:
#     print('The differences between the split ratios are not statistically significant.')





# Dataset 2
# Rows: languages, Columns: split ratios (80:20, 70:30, 60:40, 50:50)
accuracy_data = np.array([
    [0.93, 0.92, 0.91, 0.92],  # Language 1
    [0.90, 0.91, 0.90, 0.90],  # Language 2
    [0.92, 0.94, 0.94, 0.95],  # Language 3
    [0.99, 0.99, 0.99, 0.99],  # Language 4
    [0.95, 0.97, 0.96, 0.97],  # Language 5
    [0.94, 0.95, 0.94, 0.93],  # Language 6
    [0.92, 0.91, 0.92, 0.92],  # Language 7
    [0.93, 0.93, 0.91, 0.91],  # Language 8
    [0.91, 0.94, 0.94, 0.93],  # Language 9
    [0.92, 0.89, 0.91, 0.91]   # Language 10
])

# Apply the Friedman test
statistic, p_value = friedmanchisquare(accuracy_data[:, 0], accuracy_data[:, 1], accuracy_data[:, 2], accuracy_data[:, 3])

print(f'Friedman Test Statistic: {statistic}')
print(f'P-value: {p_value}')

# Interpret the result
if p_value < 0.05:
    print('The differences between the split ratios are statistically significant.')
else:
    print('The differences between the split ratios are not statistically significant.')
