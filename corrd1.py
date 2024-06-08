# -*- coding: utf-8 -*-
# @Time    : 2024/5/6 21:02
# @Author  : Chen Zean
# @Site    : 
# @File    : corrd1.py
# @Software: PyCharm
import torch

# Convert the labeled grid to a tensor
labeled_tensor = torch.tensor([
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14, 15, 16, 17, 18],
    [19, 20, 0, 0, 0, 0, 0, 21, 22],
    [23, 24, 0, 0, 0, 0, 0, 25, 26],
    [27, 28, 0, 0, 0, 0, 0, 29, 30],
    [31, 32, 0, 0, 0, 0, 0, 33, 34],
    [35, 36, 0, 0, 0, 0, 0, 37, 38],
    [39, 40, 41, 42, 43, 44, 45, 46, 47],
    [48, 49, 50, 51, 52, 53, 54, 55, 56]
])

# Function to find coordinates using PyTorch
def find_coordinates_pytorch(labeled_tensor, number):
    rows, cols = torch.where(labeled_tensor == number)
    if rows.numel() > 0 and cols.numel() > 0:
        return int(rows[0]), int(cols[0])
    else:
        print("number must less than 56")
        return None


coordinates_pytorch = {num: find_coordinates_pytorch(labeled_tensor, num) for num in range(1, 57)}
print(coordinates_pytorch)



import torch

# Define the grid size and center size
grid_size = 9
center_size = 5

# Create a 9×9 grid using PyTorch
grid_tensor = torch.arange(1, grid_size**2 + 1).reshape(grid_size, grid_size)

# Calculate the indices to remove the 5×5 center
center_start = (grid_size - center_size) // 2
center_end = center_start + center_size

# Remove the 5×5 center by setting it to 0 or another marker
grid_tensor[center_start:center_end, center_start:center_end] = 0

# Extract the remaining non-zero values and label them sequentially
remaining_indices = torch.nonzero(grid_tensor, as_tuple=True)
labeled_tensor = torch.zeros_like(grid_tensor)

# Assign sequential labels starting from 1
for idx, (row, col) in enumerate(zip(*remaining_indices), start=1):
    labeled_tensor[row, col] = idx

# Display the labeled grid
print(labeled_tensor)