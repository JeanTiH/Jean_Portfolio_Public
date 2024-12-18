'''
This code is for ACF project of QAI Inc. Fulton Maryland
It automatically identifies the largest pallet number in the database and
reads new pallet information (next N pallets) based on the current record.
--------------------
Author: JeanHan
'''
import sqlite3
import pandas as pd

'''
Set N: how many new pallets info to add
'''
N = 100

# Connect to SQLite database
conn = sqlite3.connect('ACF_database.db')
cursor = conn.cursor()

# Find the current largest PalletNo in the database
cursor.execute("SELECT PalletNo FROM pallet_data ORDER BY PalletNo DESC LIMIT 1")
largest_pallet_no = cursor.fetchone()
# Remove the 'IM' prefix and convert to an integer
largest_pallet_no = int(largest_pallet_no[0].replace('IM', '')) if largest_pallet_no else 0

# Load the workbook with the new sheets
excel_file = 'ACF (QAI to IM) Pallet Inventory.xlsx'

# Loop over new sheet numbers, starting from the next PalletNo, checking 100 new sheets
for pallet_no in range(largest_pallet_no + 1, largest_pallet_no + 1 + N):
    sheet_name = f'IM{str(pallet_no).zfill(3)}'
    try:
        # Only first 40 rows for each sheet
        df = pd.read_excel(excel_file, sheet_name=sheet_name, usecols=['QAI Pallet No', 'QAI Label', 'IM Label']).head(40)

        # Rename columns for consistency with the database
        df.columns = ['PalletNo', 'QAILabel', 'IMLabel']
        df['PalletNo'] = sheet_name  # Set PalletNo to the current sheet name (e.g., "IM417")

        # Drop rows where 'IMLabel' is missing
        df = df.dropna(subset=['QAILabel'])

        # Insert data into the SQLite table if any rows remain
        if not df.empty:
            df.to_sql('pallet_data', conn, if_exists='append', index=False)
            print(f"Data from {sheet_name} added successfully.")

    except ValueError as e:
        print(f"Sheet {sheet_name} does not exist in the workbook or encountered an error: {e}")
        break  # Stop if there are no more sheets

# Commit and close the connection
conn.commit()
conn.close()