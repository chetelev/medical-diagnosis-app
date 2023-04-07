import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()

# Generate 1000 random ages between 1 and 100
ages = np.random.randint(1, 100, size=1000)

# Generate 1000 random genders
genders = [fake.random_element(elements=('M', 'F')) for _ in range(1000)]

# Generate 1000 random values for Symptom_1 and Symptom_2
symptom_1 = [fake.random_element(
    elements=('Present', 'Absent')) for _ in range(1000)]
symptom_2 = [fake.random_element(
    elements=('Present', 'Absent')) for _ in range(1000)]

# Generate 1000 random values for Medical_History
medical_history = [fake.random_element(
    elements=('Hypertension', 'Diabetes', 'Asthma')) for _ in range(1000)]

# Generate 1000 random test results
test_results = [fake.random_element(
    elements=('Positive', 'Negative')) for _ in range(1000)]

# Generate 1000 random conditions as target variable
conditions = [fake.random_element(elements=(
    'High cholesterol', 'Heart disease', 'Arthritis')) for _ in range(1000)]

# Combine features and target variable into a single dataset
data = pd.DataFrame({'Age': ages,
                     'Gender': genders,
                     'Symptom_1': symptom_1,
                     'Symptom_2': symptom_2,
                     'Medical_History': medical_history,
                     'Test_Result': test_results,
                     'Diagnosis': conditions})

# Save the dataset
data.to_csv('synthetic_medical_data.csv', index=False)
