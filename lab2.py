import pandas as p
import numpy as np


def load_data(file_path):
    data = p.read_excel(file_path, sheet_name="Purchase data")
    return data

def create_matrix(data):
    x = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = data[["Payment (Rs)"]].values
    return x, y

def calculate_rank(x):
    return np.linalg.matrix_rank(x)

def pseudo_inverse(x):
    return np.linalg.pinv(x)

def calculate_cost(x_inverse, y):
    return x_inverse @ y


def main():

    file_path = r"C:\Users\CHANDRA SEKHAR\OneDrive\Documents\MY_SUBJECTS\ML\LAB\Lab Session Data (1).xlsx"
    data = load_data(file_path)


    x, y = create_matrix(data)
    rank = calculate_rank(x)
    x_inverse = pseudo_inverse(x)
    product_cost = calculate_cost(x_inverse, y)

    print("Matrix:")
    print(x)
    print("\nOutput:")
    print(y)

    print("Rank:",rank)

    print("\nCost of Each Product")
    print("Candies:",product_cost[0][0])
    print("Mangoes:",product_cost[1][0])
    print("Milk Packets:",product_cost[2][0])

if __name__ == "__main__":
    main()