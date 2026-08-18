import pandas as pd
import numpy as np


# ---------------------------------------------------------
# 1. Encode categorical columns
# ---------------------------------------------------------
def encode_categorical_data(data):
    data = data.copy()

    categorical_columns = data.select_dtypes(include=["object"]).columns

    for column in categorical_columns:
        data[column] = data[column].fillna(
            data[column].mode()[0]
        )

        categories = data[column].unique()
        category_map = {value: index for index, value in enumerate(categories)}

        data[column] = data[column].map(category_map)

    return data


# ---------------------------------------------------------
# 2. Fill missing numerical values
# ---------------------------------------------------------
def impute_missing_values(data, method="median"):
    data = data.copy()

    numerical_columns = data.select_dtypes(include=np.number).columns

    for column in numerical_columns:

        if method == "mean":
            replacement = data[column].mean()

        elif method == "median":
            replacement = data[column].median()

        elif method == "mode":
            replacement = data[column].mode()[0]

        else:
            raise ValueError("Choose mean, median or mode.")

        data[column] = data[column].fillna(replacement)

    return data


# ---------------------------------------------------------
# 3. Calculate Euclidean distance
# ---------------------------------------------------------
def calculate_distance(point1, point2):
    distance = 0

    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2

    return np.sqrt(distance)


# ---------------------------------------------------------
# 4. Bubble sort
# ---------------------------------------------------------
def bubble_sort(distances):
    distances = distances.copy()

    for i in range(len(distances)):
        for j in range(0, len(distances) - i - 1):

            if distances[j][0] > distances[j + 1][0]:
                distances[j], distances[j + 1] = (
                    distances[j + 1],
                    distances[j]
                )

    return distances


# ---------------------------------------------------------
# 5. Selection sort
# ---------------------------------------------------------
def selection_sort(distances):
    distances = distances.copy()

    for i in range(len(distances)):
        smallest = i

        for j in range(i + 1, len(distances)):
            if distances[j][0] < distances[smallest][0]:
                smallest = j

        distances[i], distances[smallest] = (
            distances[smallest],
            distances[i]
        )

    return distances


# ---------------------------------------------------------
# 6. Insertion sort
# ---------------------------------------------------------
def insertion_sort(distances):
    distances = distances.copy()

    for i in range(1, len(distances)):
        current = distances[i]
        j = i - 1

        while j >= 0 and distances[j][0] > current[0]:
            distances[j + 1] = distances[j]
            j -= 1

        distances[j + 1] = current

    return distances


# ---------------------------------------------------------
# 7. Select sorting algorithm
# ---------------------------------------------------------
def sort_distances(distances, sorting_method="bubble"):
    if sorting_method == "bubble":
        return bubble_sort(distances)

    elif sorting_method == "selection":
        return selection_sort(distances)

    elif sorting_method == "insertion":
        return insertion_sort(distances)

    else:
        raise ValueError(
            "Sorting method must be bubble, selection or insertion."
        )


# ---------------------------------------------------------
# 8. Find k nearest neighbours
# ---------------------------------------------------------
def find_neighbours(training_data, training_labels, test_point,
                    k=3, sorting_method="bubble"):

    distances = []

    for index in range(len(training_data)):

        distance = calculate_distance(
            training_data[index],
            test_point
        )

        # Index is used as a tie breaker when distances are equal
        distances.append(
            (distance, index, training_labels[index])
        )

    distances = sort_distances(
        distances,
        sorting_method
    )

    return distances[:k]


# ---------------------------------------------------------
# 9. Majority voting with tie breaking
# ---------------------------------------------------------
def assign_class(neighbours):
    class_votes = {}

    for distance, index, class_label in neighbours:

        if class_label not in class_votes:
            class_votes[class_label] = 0

        class_votes[class_label] += 1

    maximum_votes = max(class_votes.values())

    possible_classes = []

    for class_label, votes in class_votes.items():
        if votes == maximum_votes:
            possible_classes.append(class_label)

    # Tie-breaking: choose the class of the closest neighbour
    if len(possible_classes) > 1:
        for distance, index, class_label in neighbours:
            if class_label in possible_classes:
                return class_label

    return possible_classes[0]


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

# Load the project dataset

data = pd.read_excel(
    r"C:\Users\CHANDRA SEKHAR\OneDrive\Documents\MY_SUBJECTS\ML\LAB\Lab Session Data (1).xlsx",
    sheet_name="marketing_campaign"
)

# Remove ID because it does not represent a useful feature
data = data.drop(columns=["ID"])

# Separate input features and target
X = data.drop(columns=["Response"])
y = data["Response"]

# Encode categorical columns
X = encode_categorical_data(X)

# Convert everything to numeric where possible
X = X.apply(pd.to_numeric, errors="coerce")

# Fill missing numerical values
X = impute_missing_values(X, method="median")

# Convert data into arrays
X_values = X.to_numpy(dtype=float)
y_values = y.to_numpy()

# Use the first 80% as training data
split_point = int(len(X_values) * 0.8)

X_train = X_values[:split_point]
y_train = y_values[:split_point]

X_test = X_values[split_point:]
y_test = y_values[split_point:]

# Select one test pattern
test_pattern = X_test[0]

# Find 3 nearest neighbours
nearest_neighbours = find_neighbours(
    X_train,
    y_train,
    test_pattern,
    k=3,
    sorting_method="bubble"
)

# Predict the class
predicted_class = assign_class(nearest_neighbours)

print("Nearest Neighbours:")
for distance, index, class_label in nearest_neighbours:
    print(
        "Distance:", round(distance, 3),
        "Class:", class_label
    )

print("\nActual Class:", y_test[0])
print("Predicted Class:", predicted_class)