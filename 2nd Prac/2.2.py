# Import required libraries
import pandas as pd
import pyodbc
import cx_Oracle
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------
# ETL PROCESS
# ----------------------

# 1. EXTRACT: Get data from Excel
def extract_excel(file_path):
    """Extracts data from Excel file
    Args:
        file_path (str): Path to Excel file
    Returns:
        DataFrame: Pandas DataFrame with extracted data
    """
    return pd.read_excel(file_path)

# 2. EXTRACT: Get data from SQL Server
def extract_sql(server, db, query):
    """Extracts data from SQL Server
    Args:
        server (str): Server name/address
        db (str): Database name
        query (str): SQL query to execute
    Returns:
        DataFrame: Resultset as DataFrame
    """
    conn = pyodbc.connect(
        f'DRIVER={{SQL Server}};SERVER={server};DATABASE={db};Trusted_Connection=yes;'
    )
    data = pd.read_sql(query, conn)
    conn.close()
    return data

# 3. EXTRACT: Get data from Oracle
def extract_oracle(user, pwd, dsn, query):
    """Extracts data from Oracle DB
    Args:
        user (str): Database username
        pwd (str): Database password
        dsn (str): Data Source Name
        query (str): SQL query
    Returns:
        DataFrame: Query results as DataFrame
    """
    conn = cx_Oracle.connect(user, pwd, dsn)
    data = pd.read_sql(query, conn)
    conn.close()
    return data

# 4. TRANSFORM: Clean and merge data
def transform_data(*dfs):
    """Transforms multiple DataFrames
    Args:
        *dfs: Variable number of DataFrames
    Returns:
        DataFrame: Cleaned, merged DataFrame
    """
    # Concatenate data vertically
    combined = pd.concat(dfs, ignore_index=True)
    
    # Data cleaning
    combined.dropna(inplace=True)  # Remove missing values
    combined.drop_duplicates(inplace=True)  # Remove duplicates
    
    # Convert date columns (example)
    if 'order_date' in combined.columns:
        combined['order_date'] = pd.to_datetime(combined['order_date'])
    
    return combined

# 5. LOAD: Save to target system
def load_data(df, target='output.csv'):
    """Loads data to target destination
    Args:
        df (DataFrame): Processed data
        target (str): Output file/database
    """
    # Save to CSV for this example
    df.to_csv(target, index=False)
    
# ----------------------
# DATA VISUALIZATION
# ----------------------
def visualize_data(df):
    """Creates visualizations from processed data
    Args:
        df (DataFrame): Processed BI data
    """
    plt.figure(figsize=(12, 6))
    
    # Sales Trend Visualization
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df, x='order_date', y='sales', estimator='sum')
    plt.title('Monthly Sales Trend')
    plt.xticks(rotation=45)
    
    # Product Performance Visualization
    plt.subplot(1, 2, 2)
    sns.barplot(data=df, x='product_category', y='sales', estimator=sum)
    plt.title('Sales by Product Category')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('sales_analysis.png')
    plt.show()

# ----------------------
# MAIN WORKFLOW
# ----------------------
def main():
    try:
        # EXTRACT PHASE
        excel_data = extract_excel('sales_data.xlsx')
        sql_data = extract_sql('localhost', 'SalesDB', 'SELECT * FROM orders')
        oracle_data = extract_oracle('bi_user', 'password123', 'localhost/ORCL', 
                                   'SELECT * FROM product_sales')
        
        # TRANSFORM PHASE
        cleaned_data = transform_data(excel_data, sql_data, oracle_data)
        
        # LOAD PHASE
        load_data(cleaned_data, 'bi_output.csv')
        
        # VISUALIZATION PHASE
        visualize_data(cleaned_data)
        
    except Exception as e:
        print(f"ETL Process Failed: {str(e)}")

if __name__ == "__main__":
    main()