import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
# Replace with your actual file path
file_path = 'runs/classify/train62/results.csv'
data = pd.read_csv(file_path)

print("Column names:", data.columns)

# Check the first few rows of the data to ensure it has the correct format
print(data.head())

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(data['                  epoch'],
         data['             train/loss'], label='Training Loss')
plt.plot(data['                  epoch'],
         data['               val/loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()
