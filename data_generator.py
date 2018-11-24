import random
import pandas as pd

STATES = ["Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa","Gujarat","Haryana","Himachal Pradesh","Jammu and Kashmir","Jharkhand","Karnataka","Kerala","Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland","Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura","Uttar Pradesh","Uttarakhand","West Bengal"]
EPC_VENDORS = ["LNT", "Nagarjuna", "GMR", "Tata", "Gammon"]

COLUMNS = ["Geography", "EPC Vendors", "Risk", "Historical Quote", "Bid Success"]

rows = 2000
state_column = [random.choice(STATES) for i in range(rows)]
vendor_column = [random.choice(EPC_VENDORS) for i in range(rows)]
risk_column = [random.randint(5, 90) for i in range(rows)]
bid_price = [random.randint(25000, 50000) for i in range(rows)]
success_column = [random.choice([0, 1]) for i in range(rows)]

df = pd.DataFrame({
	COLUMNS[0]: state_column,
	COLUMNS[1]: vendor_column,
	COLUMNS[2]: risk_column,
	COLUMNS[3]: bid_price,
	COLUMNS[4]: success_column
	})

print(df.head())
df.to_csv(path_or_buf="data3.csv")