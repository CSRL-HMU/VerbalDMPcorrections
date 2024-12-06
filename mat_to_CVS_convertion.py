from scipy.io import loadmat
import pandas as pd

# Load the .mat file
mat_file_path = '/Users/elenikonstantinidou/Desktop/diplwmatiki/diplwmatiki codes/execution Robot/twest_demo.mat'
mat_data = loadmat(mat_file_path)

# Extract 't' and 'p' variables from the .mat file
t = mat_data['t'].flatten()  # Convert 't' to 1D array
p = mat_data['p']  # 'p' is already in shape (2555, 3)
x = mat_data['x']  # 'x' is already in shape (2555, 7)

# Create a DataFrame with 't' and columns from 'p'
df = pd.DataFrame({
    't': t,
    'p_x': p[:, 0],  # First column of p
    'p_y': p[:, 1],  # Second column of p
    'p_z': p[:, 2]   # Third column of p
})

df_2 = pd.DataFrame({
    't': t,
    'x_1': x[:, 0],  # First column of x
    'x_2': x[:, 1],  # Second column of x   
    'x_3': x[:, 2],  # Third column of x
    'x_4': x[:, 3],  # Fourth column of x
    'x_5': x[:, 4],  # Fifth column of x
    'x_6': x[:, 5],  # Sixth column of x
    'x_7': x[:, 6]   # Seventh column of x
}) 

# Save the DataFrame to a CSV file
csv_file_path = '/Users/elenikonstantinidou/Desktop/test_motion_demo_p.csv'
csv_file_path_2 = '/Users/elenikonstantinidou/Desktop/test_motion_demo_x.csv'

df.to_csv(csv_file_path, index=False)
df_2.to_csv(csv_file_path_2, index=False)

print(f"\nFile saved as {csv_file_path}")
print(f"\nFile saved as {csv_file_path_2}")