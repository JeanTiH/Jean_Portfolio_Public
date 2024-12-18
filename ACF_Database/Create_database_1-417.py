'''
This code is for ACF project of QAI Inc. Fulton, Maryland
It reads data from an Excel file and creates an SQLite database.
--------------------
Author: JeanHan
'''
import sqlite3
import pandas as pd

'''
Read the history data Pallet001-Pallet417
'''
# Connect to SQLite database
conn = sqlite3.connect('ACF_database(1-417).db')
cursor = conn.cursor()

# Create the table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS pallet_data (
    PalletNo TEXT,
    QAILabel TEXT,
    IMLabel TEXT
)
''')
conn.commit()

# Load the workbook
excel_file = 'ACF (QAI to IM) Pallet Inventory.xlsx'
sheet_names = [f'IM{str(i).zfill(3)}' for i in range(1, 418)]  # IM001 to IM417

for sheet in sheet_names:
    # Only first 40 rows for each sheet
    df = pd.read_excel(excel_file, sheet_name=sheet, usecols=['QAI Pallet No', 'QAI Label', 'IM Label']).head(40)

    # Rename columns for consistency with the database
    df.columns = ['PalletNo', 'QAILabel', 'IMLabel']

    # Drop rows where 'IMLabel' is missing
    df = df.dropna(subset=['QAILabel'])

    # Insert data into the SQLite table if any rows remain
    if not df.empty:
        df.to_sql('pallet_data', conn, if_exists='append', index=False)

# Commit and close the connection
conn.commit()
conn.close()