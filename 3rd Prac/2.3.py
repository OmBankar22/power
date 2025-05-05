# Import libraries
import pandas as pd
from sqlalchemy import create_engine

# ----------------------------------------
# Step 1: Extract Data from Sources
# ----------------------------------------

# Extract from Excel (Sales)
sales_df = pd.read_excel("sales_data.xlsx", sheet_name="Sales")
print("Raw Sales Data:")
print(sales_df.head())

# Extract from CSV (Customers)
customers_df = pd.read_csv("customers.csv")
print("\nRaw Customer Data:")
print(customers_df.head())

# Extract from Excel (Products)
products_df = pd.read_excel("products.xlsx", sheet_name="Products")
print("\nRaw Product Data:")
print(products_df.head())

# ----------------------------------------
# Step 2: Transform Data
# ----------------------------------------

# Merge Sales with Customer Data
merged_df = pd.merge(sales_df, customers_df, on="CustomerID", how="left")

# Merge with Product Data
final_df = pd.merge(merged_df, products_df, on="Product", how="left")

# Calculate Total Revenue per Order
final_df["TotalRevenue"] = final_df["Quantity"] * final_df["Revenue"]

# Handle Missing Values (if any)
final_df.dropna(inplace=True)

print("\nTransformed Data:")
print(final_df.head())

# ----------------------------------------
# Step 3: Load Data to SQL Server
# ----------------------------------------

# Create SQL Server Connection
server = "localhost"
database = "BI_Database"
username = "sa"
password = "your_password"

# Connection String
engine = create_engine(
    f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
)

# Load Data to SQL Server
final_df.to_sql("SalesReport", engine, if_exists="replace", index=False)
print("\nData loaded to SQL Server!")

# ----------------------------------------
# Step 4: Connect Power BI to SQL Server
# ----------------------------------------
# (Demonstrate in Power BI Desktop)
# 1. Open Power BI > Get Data > SQL Server.
# 2. Enter server name: localhost
# 3. Select database: BI_Database
# 4. Load the "SalesReport" table.