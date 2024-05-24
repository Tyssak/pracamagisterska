import re

# Define a regular expression pattern to match validation accuracy lines
pattern = r"val_accuracy: ([0-9.]+)"

# Open the file
with open("output0.txt", "r") as file:
    content = file.read()

# Find all matches of validation accuracy
matches = re.findall(pattern, content)

# Convert the accuracy values to floats
accuracy_values = [float(match) for match in matches]

# Find the highest accuracy value and its corresponding index
highest_accuracy = max(accuracy_values)
index_of_highest_accuracy = accuracy_values.index(highest_accuracy)

# Print the highest accuracy value and its index
print("Highest validation accuracy:", highest_accuracy)
#print("Index of highest accuracy value:", index_of_highest_accuracy)
