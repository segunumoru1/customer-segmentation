import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

data = {
    'annual_income': np.random.randint(20000, 150000, 100),
    'spending_score': np.random.randint(1, 100, 100),
    'savings': np.random.randint(1000, 50000, 100),
    'age': np.random.randint(18, 70, 100)
}

df = pd.DataFrame(data)
df.to_csv('sample_customers.csv', index=False)
print("Sample data saved as 'sample_customers.csv'")