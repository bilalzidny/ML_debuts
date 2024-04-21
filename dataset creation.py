import pandas as pd
import random as rd

# Generate data
x = [i for i in range(700)]
y = [rd.normalvariate(x[i], 200) for i in range(700)]

# Create a DataFrame
df = pd.DataFrame({'x': x, 'y': y})

# Save the DataFrame to a CSV file
df.to_csv('data.csv', index=False)


