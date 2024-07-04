from scipy.stats import wilcoxon

# Example accuracy data for different train-test split ratios
accuracies_80_20 = [0.99, 0.99, 0.98, 1.00, 0.97, 1.00, 0.99, 0.97]  # accuracies for 80:20 split
accuracies_70_30 = [0.98, 0.98, 0.98, 1.00, 0.94, 1.00, 0.99, 0.94]  # accuracies for 70:30 split
accuracies_60_40 = [0.99, 0.98, 0.97, 1.00, 0.96, 1.00, 0.98, 0.95]  # accuracies for 60:40 split
accuracies_50_50 = [0.99, 0.99, 0.98, 1.00, 0.94, 1.00, 0.98, 0.92]  # accuracies for 50:50 split

# Perform Wilcoxon signed-rank test between the different split ratios
stat_80_20_vs_70_30, p_value_80_20_vs_70_30 = wilcoxon(accuracies_80_20, accuracies_70_30)
print(f'Wilcoxon Test between 80:20 and 70:30: Statistic = {stat_80_20_vs_70_30}, p-value = {p_value_80_20_vs_70_30}')

stat_80_20_vs_60_40, p_value_80_20_vs_60_40 = wilcoxon(accuracies_80_20, accuracies_60_40)
print(f'Wilcoxon Test between 80:20 and 60:40: Statistic = {stat_80_20_vs_60_40}, p-value = {p_value_80_20_vs_60_40}')

stat_80_20_vs_50_50, p_value_80_20_vs_50_50 = wilcoxon(accuracies_80_20, accuracies_50_50)
print(f'Wilcoxon Test between 80:20 and 50:50: Statistic = {stat_80_20_vs_50_50}, p-value = {p_value_80_20_vs_50_50}')

stat_70_30_vs_60_40, p_value_70_30_vs_60_40 = wilcoxon(accuracies_70_30, accuracies_60_40)
print(f'Wilcoxon Test between 70:30 and 60:40: Statistic = {stat_70_30_vs_60_40}, p-value = {p_value_70_30_vs_60_40}')

stat_70_30_vs_50_50, p_value_70_30_vs_50_50 = wilcoxon(accuracies_70_30, accuracies_50_50)
print(f'Wilcoxon Test between 70:30 and 50:50: Statistic = {stat_70_30_vs_50_50}, p-value = {p_value_70_30_vs_50_50}')

stat_60_40_vs_50_50, p_value_60_40_vs_50_50 = wilcoxon(accuracies_60_40, accuracies_50_50)
print(f'Wilcoxon Test between 60:40 and 50:50: Statistic = {stat_60_40_vs_50_50}, p-value = {p_value_60_40_vs_50_50}')

# Interpret the result
alpha = 0.05
if p_value_80_20_vs_70_30 < alpha:
    print(f'The difference between 80:20 and 70:30 split ratios is statistically significant (p = {p_value_80_20_vs_70_30:.5f})')
else:
    print(f'The difference between 80:20 and 70:30 split ratios is not statistically significant (p = {p_value_80_20_vs_70_30:.5f})')

if p_value_80_20_vs_60_40 < alpha:
    print(f'The difference between 80:20 and 60:40 split ratios is statistically significant (p = {p_value_80_20_vs_60_40:.5f})')
else:
    print(f'The difference between 80:20 and 60:40 split ratios is not statistically significant (p = {p_value_80_20_vs_60_40:.5f})')

if p_value_80_20_vs_50_50 < alpha:
    print(f'The difference between 80:20 and 50:50 split ratios is statistically significant (p = {p_value_80_20_vs_50_50:.5f})')
else:
    print(f'The difference between 80:20 and 50:50 split ratios is not statistically significant (p = {p_value_80_20_vs_50_50:.5f})')

if p_value_70_30_vs_60_40 < alpha:
    print(f'The difference between 70:30 and 60:40 split ratios is statistically significant (p = {p_value_70_30_vs_60_40:.5f})')
else:
    print(f'The difference between 70:30 and 60:40 split ratios is not statistically significant (p = {p_value_70_30_vs_60_40:.5f})')

if p_value_70_30_vs_50_50 < alpha:
    print(f'The difference between 70:30 and 50:50 split ratios is statistically significant (p = {p_value_70_30_vs_50_50:.5f})')
else:
    print(f'The difference between 70:30 and 50:50 split ratios is not statistically significant (p = {p_value_70_30_vs_50_50:.5f})')

if p_value_60_40_vs_50_50 < alpha:
    print(f'The difference between 60:40 and 50:50 split ratios is statistically significant (p = {p_value_60_40_vs_50_50:.5f})')
else:
    print(f'The difference between 60:40 and 50:50 split ratios is not statistically significant (p = {p_value_60_40_vs_50_50:.5f})')









# Dataset 2
# Example accuracy data for different train-test split ratios
# accuracies_80_20 = [0.93, 0.90, 0.92, 0.99, 0.95, 0.94, 0.92, 0.93, 0.91, 0.92]  # accuracies for 80:20 split
# accuracies_70_30 = [0.92, 0.91, 0.94, 0.99, 0.97, 0.95, 0.91, 0.93, 0.94, 0.89]  # accuracies for 70:30 split
# accuracies_60_40 = [0.91, 0.90, 0.94, 0.99, 0.96, 0.94, 0.92, 0.91, 0.94, 0.91]  # accuracies for 60:40 split
# accuracies_50_50 = [0.92, 0.90, 0.95, 0.99, 0.97, 0.93, 0.92, 0.91, 0.93, 0.91]  # accuracies for 50:50 split

# stat_80_20_vs_70_30, p_value_80_20_vs_70_30 = wilcoxon(accuracies_80_20, accuracies_70_30)
# print(f'Wilcoxon Test between 80:20 and 70:30: Statistic = {stat_80_20_vs_70_30}, p-value = {p_value_80_20_vs_70_30}')

# # Perform Wilcoxon signed-rank test between 80:20 and 60:40 split ratios
# stat_80_20_vs_60_40, p_value_80_20_vs_60_40 = wilcoxon(accuracies_80_20, accuracies_60_40)
# print(f'Wilcoxon Test between 80:20 and 60:40: Statistic = {stat_80_20_vs_60_40}, p-value = {p_value_80_20_vs_60_40}')

# # Perform Wilcoxon signed-rank test between 80:20 and 50:50 split ratios
# stat_80_20_vs_50_50, p_value_80_20_vs_50_50 = wilcoxon(accuracies_80_20, accuracies_50_50)
# print(f'Wilcoxon Test between 80:20 and 50:50: Statistic = {stat_80_20_vs_50_50}, p-value = {p_value_80_20_vs_50_50}')

# stat_70_30_vs_60_40, p_value_70_30_vs_60_40 = wilcoxon(accuracies_70_30, accuracies_60_40)
# print(f'Wilcoxon Test between 70:30 and 60:40: Statistic = {stat_70_30_vs_60_40}, p-value = {p_value_70_30_vs_60_40}')

# stat_70_30_vs_50_50, p_value_70_30_vs_50_50 = wilcoxon(accuracies_70_30, accuracies_50_50)
# print(f'Wilcoxon Test between 70:30 and 50:50: Statistic = {stat_70_30_vs_50_50}, p-value = {p_value_70_30_vs_50_50}')

# # Perform Wilcoxon signed-rank test between 60:40 and 50:50 split ratios
# stat_60_40_vs_50_50, p_value_60_40_vs_50_50 = wilcoxon(accuracies_60_40, accuracies_50_50)
# print(f'Wilcoxon Test between 60:40 and 50:50: Statistic = {stat_60_40_vs_50_50}, p-value = {p_value_60_40_vs_50_50}')

# # Interpret the result
# alpha = 0.05
# if p_value_80_20_vs_70_30 < alpha:
#     print(f'The difference between 80:20 and 70:30 split ratios is statistically significant (p = {p_value_80_20_vs_70_30:.5f})')
# else:
#     print(f'The difference between 80:20 and 70:30 split ratios is not statistically significant (p = {p_value_80_20_vs_70_30:.5f})')

# if p_value_80_20_vs_60_40 < alpha:
#     print(f'The difference between 80:20 and 60:40 split ratios is statistically significant (p = {p_value_80_20_vs_60_40:.5f})')
# else:
#     print(f'The difference between 80:20 and 60:40 split ratios is not statistically significant (p = {p_value_80_20_vs_60_40:.5f})')

# if p_value_80_20_vs_50_50 < alpha:
#     print(f'The difference between 80:20 and 50:50 split ratios is statistically significant (p = {p_value_80_20_vs_50_50:.5f})')
# else:
#     print(f'The difference between 80:20 and 50:50 split ratios is not statistically significant (p = {p_value_80_20_vs_50_50:.5f})')

# if p_value_70_30_vs_60_40 < alpha:
#     print(f'The difference between 70:30 and 60:40 split ratios is statistically significant (p = {p_value_70_30_vs_60_40:.5f})')
# else:
#     print(f'The difference between 70:30 and 60:40 split ratios is not statistically significant (p = {p_value_70_30_vs_60_40:.5f})')

# if p_value_70_30_vs_50_50 < alpha:
#     print(f'The difference between 70:30 and 50:50 split ratios is statistically significant (p = {p_value_70_30_vs_50_50:.5f})')
# else:
#     print(f'The difference between 70:30 and 50:50 split ratios is not statistically significant (p = {p_value_70_30_vs_50_50:.5f})')

# if p_value_60_40_vs_50_50 < alpha:
#     print(f'The difference between 60:40 and 50:50 split ratios is statistically significant (p = {p_value_60_40_vs_50_50:.5f})')
# else:
#     print(f'The difference between 60:40 and 50:50 split ratios is not statistically significant (p = {p_value_60_40_vs_50_50:.5f})')
