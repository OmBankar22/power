# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------
# Step 1: Load Excel Data
# ----------------------------------------

# Read all sheets from Excel
excel_file = "/content/sales_data.xlsx"
sales_df = pd.read_excel(excel_file, sheet_name="Sales")
customers_df = pd.read_excel(excel_file, sheet_name="Customers")
products_df = pd.read_excel(excel_file, sheet_name="Products")

print("Sales Data:")
display(sales_df.head())

print("\nCustomers Data:")
display(customers_df.head())

print("\nProducts Data:")
display(products_df.head())

# ----------------------------------------
# Step 2: Data Cleaning (Excel-like "Format as Table")
# ----------------------------------------

# Remove duplicates
sales_df.drop_duplicates(inplace=True)

# Handle missing values (e.g., fill with 0)
sales_df["Quantity"] = sales_df["Quantity"].fillna(0)

# Convert OrderDate to datetime
sales_df["OrderDate"] = pd.to_datetime(sales_df["OrderDate"])

# ----------------------------------------
# Step 3: Advanced Excel Formulas (VLOOKUP, SUMIF)
# ----------------------------------------

# Merge Customer Name and Region into Sales Data
sales_df = pd.merge(sales_df, customers_df[["CustomerID", "CustomerName", "Region"]], on="CustomerID", how="left")

# Rename Price column in Products to avoid duplicate conflict
products_df = products_df.rename(columns={"Price": "ProductPrice"})

# Merge Product Price into Sales Data
sales_df = pd.merge(sales_df, products_df[["Product", "ProductPrice"]], on="Product", how="left")

# Calculate Total Revenue (Quantity * ProductPrice)
sales_df["TotalRevenue"] = sales_df["Quantity"] * sales_df["ProductPrice"]

print("\nMerged Data with Revenue:")
display(sales_df.head())

# ----------------------------------------
# Step 4: PivotTables (Excel-like Summarization)
# ----------------------------------------
print("Columns available for Pivot Table:", sales_df.columns)
# Use correct Region column name
region_col = "Region"
if region_col not in sales_df.columns:
    region_col = "Region_x" if "Region_x" in sales_df.columns else "Region_y" if "Region_y" in sales_df.columns else None

if region_col is None:
    raise ValueError("No 'Region' column found for PivotTable. Check merged data.")

# Create a PivotTable: Total Revenue by Region and Product
pivot_table = pd.pivot_table(
    sales_df,
    values="TotalRevenue",
    index=region_col,
    columns="Product",
    aggfunc="sum",
    fill_value=0
)

# # Create a PivotTable: Total Revenue by Region and Product
# pivot_table = pd.pivot_table(
#     sales_df,
#     values="TotalRevenue",
#     index="Region",
#     columns="Product",
#     aggfunc="sum",
#     fill_value=0
# )

print("\nPivotTable (Revenue by Region & Product):")
display(pivot_table)

# ----------------------------------------
# Step 5: Visualization (Excel-like Charts)
# ----------------------------------------

# Bar Chart: Total Revenue by Product
plt.figure(figsize=(10, 6))
sns.barplot(data=sales_df, x="Product", y="TotalRevenue", estimator=sum, ci=None)
plt.title("Total Revenue by Product (Excel-like Bar Chart)")
plt.xlabel("Product")
plt.ylabel("Total Revenue ($)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Line Chart: Monthly Sales Trend
sales_df["Month"] = sales_df["OrderDate"].dt.month_name()
monthly_sales = sales_df.groupby("Month")["TotalRevenue"].sum().reset_index()

# Sort months in calendar order
from pandas.tseries.offsets import MonthBegin
monthly_sales["MonthNumber"] = pd.to_datetime(monthly_sales["Month"], format='%B') + MonthBegin(0)
monthly_sales = monthly_sales.sort_values("MonthNumber")

plt.figure(figsize=(10, 6))
sns.lineplot(data=monthly_sales, x="Month", y="TotalRevenue", marker="o")
plt.title("Monthly Sales Trend (Excel-like Line Chart)")
plt.xlabel("Month")
plt.ylabel("Total Revenue ($)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------------------
# Step 6: Export Results to Excel
# ----------------------------------------

with pd.ExcelWriter("analysis_results.xlsx") as writer:
    sales_df.to_excel(writer, sheet_name="Processed Sales", index=False)
    pivot_table.to_excel(writer, sheet_name="PivotTable")
