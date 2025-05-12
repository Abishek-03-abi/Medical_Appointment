import pandas as pd
import numpy as np

raw_df = pd.read_csv('data/raw/medical_appointments_raw.csv')
df = raw_df.copy()
numeric_cols = ['Age', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism']
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

categorical_cols = ['Gender', 'Neighbourhood', 'SMS_received']
df[categorical_cols] = df[categorical_cols].fillna('Unknown')

df.drop_duplicates(subset=['PatientId', 'AppointmentID'], keep='first', inplace=True)
df['Gender'] = df['Gender'].str.upper().str[0]

df['No-show'] = df['No-show'].map({'Yes': 1, 'No': 0})
df['Age'] = df['Age'].clip(lower=0, upper=110)

date_cols = ['ScheduledDay', 'AppointmentDay']
df[date_cols] = df[date_cols].apply(lambda x: pd.to_datetime(x).dt.date)
df.columns = df.columns.str.lower().str.replace('-', '_')
df.rename(columns={'no_show': 'missed_appointment'}, inplace=True)
df.to_csv('data/processed/medical_appointments_clean.csv', index=False)

print("Cleaning complete!")
print(f"Original shape: {raw_df.shape} | Cleaned shape: {df.shape}")
print(f"Duplicates removed: {len(raw_df) - len(df)}")
