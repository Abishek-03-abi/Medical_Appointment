import pandas as pd
import numpy as np

# Load raw data (preserve original)
raw_df = pd.read_csv('data/raw/medical_appointments_raw.csv')
df = raw_df.copy()  # Work on a copy

# ======================
# 1. Handle Missing Values
# ======================
# Fill missing numeric values with median
numeric_cols = ['Age', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism']
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill missing categorical values
categorical_cols = ['Gender', 'Neighbourhood', 'SMS_received']
df[categorical_cols] = df[categorical_cols].fillna('Unknown')

# ======================
# 2. Remove Duplicates
# ======================
df.drop_duplicates(subset=['PatientId', 'AppointmentID'], keep='first', inplace=True)

# ======================
# 3. Standardize Values
# ======================
# Gender: M/F
df['Gender'] = df['Gender'].str.upper().str[0]

# No-show: Convert to binary (1 for missed, 0 for attended)
df['No-show'] = df['No-show'].map({'Yes': 1, 'No': 0})

# ======================
# 4. Clean Age Column
# ======================
# Remove negative ages and cap at 110
df['Age'] = df['Age'].clip(lower=0, upper=110)

# ======================
# 5. Fix Date Columns
# ======================
date_cols = ['ScheduledDay', 'AppointmentDay']
df[date_cols] = df[date_cols].apply(lambda x: pd.to_datetime(x).dt.date)

# ======================
# 6. Rename Columns
# ======================
df.columns = df.columns.str.lower().str.replace('-', '_')
df.rename(columns={'no_show': 'missed_appointment'}, inplace=True)

# ======================
# 7. Save Cleaned Data
# ======================
df.to_csv('data/processed/medical_appointments_clean.csv', index=False)

print("Cleaning complete!")
print(f"Original shape: {raw_df.shape} | Cleaned shape: {df.shape}")
print(f"Duplicates removed: {len(raw_df) - len(df)}")