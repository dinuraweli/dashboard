# historical_import.py
import pandas as pd
import sqlite3
import os
import glob
from datetime import datetime
import re

# Import your existing functions from dashboard
from dashboard import PortfolioHistory, clean_numeric

def process_historical_file(df):
    """Process a single historical Excel file with the specific format"""
    
    # Remove first 2 rows (header rows)
    df_clean = df.iloc[2:].copy()
    
    # Remove the last row if it contains "Total"
    if len(df_clean) > 0 and 'Total' in str(df_clean.iloc[-1, 0]):
        df_clean = df_clean.iloc[:-1]
    
    # Reset index
    df_clean.reset_index(drop=True, inplace=True)
    
    # Define column names based on your structure
    columns = [
        'Security', 'Quantity', 'Cleared Balance', 'Available Balance', 
        'Unsettled Buy', 'Unsettled Sell', 'Holding % (Quantity)', 
        'Avg Price', 'BES Price', 'Total Cost', 'Traded Price', 
        'Market Value', 'Holding % (Market Value)', 'Sales Commission',
        'Sales Proceeds', 'Unrealized Gain / (Loss)', 'Unrealized Gain/Loss %',
        'Unr Today Gain/(Loss)'
    ]
    
    # Assign column names
    if len(df_clean.columns) >= len(columns):
        df_clean.columns = columns[:len(df_clean.columns)]
    else:
        df_clean.columns = [f'Column_{i}' for i in range(len(df_clean.columns))]
    
    # Clean numeric columns
    numeric_cols = [
        'Quantity', 'Holding % (Quantity)', 'Avg Price', 'BES Price', 
        'Total Cost', 'Traded Price', 'Market Value', 'Holding % (Market Value)',
        'Sales Commission', 'Sales Proceeds', 'Unrealized Gain / (Loss)',
        'Unrealized Gain/Loss %', 'Unr Today Gain/(Loss)'
    ]
    
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(clean_numeric)
    
    # Convert Quantity to integer
    if 'Quantity' in df_clean.columns:
        df_clean['Quantity'] = df_clean['Quantity'].fillna(0).astype(int)
    
    # Calculate additional metrics
    if all(col in df_clean.columns for col in ['Total Cost', 'Quantity', 'Market Value']):
        df_clean['Cost per Share'] = df_clean.apply(
            lambda row: row['Total Cost'] / row['Quantity'] if row['Quantity'] > 0 else 0, 
            axis=1
        )
        df_clean['Current Price'] = df_clean.apply(
            lambda row: row['Market Value'] / row['Quantity'] if row['Quantity'] > 0 else 0, 
            axis=1
        )
    
    if 'Unrealized Gain / (Loss)' in df_clean.columns and 'Total Cost' in df_clean.columns:
        df_clean['Return %'] = df_clean.apply(
            lambda row: (row['Unrealized Gain / (Loss)'] / row['Total Cost'] * 100) if row['Total Cost'] != 0 else 0,
            axis=1
        )
    
    return df_clean

def import_historical_files(folder_path, file_pattern="*.xlsx"):
    """Import all historical portfolio files into the database"""
    
    # Initialize history database
    history_db = PortfolioHistory()
    
    # Get all Excel files in the folder
    excel_files = glob.glob(os.path.join(folder_path, file_pattern))
    
    print(f"Found {len(excel_files)} Excel files to process")
    
    # Sort files to process in chronological order
    excel_files.sort()
    
    success_count = 0
    error_count = 0
    duplicate_count = 0
    
    for file_path in excel_files:
        try:
            # Extract date from filename
            filename = os.path.basename(file_path)
            
            # Look for YYYY-MM-DD pattern in filename
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
            
            if not date_match:
                print(f"⚠️ Skipping {filename}: No date found in filename")
                error_count += 1
                continue
            
            file_date = date_match.group(1)
            
            print(f"\n📄 Processing: {filename} → Date: {file_date}")
            
            # Read the raw Excel file (no header)
            df_raw = pd.read_excel(file_path, header=None)
            
            print(f"   Raw shape: {df_raw.shape}")
            print(f"   First row: {df_raw.iloc[0, 0] if df_raw.shape[0] > 0 else 'Empty'}")
            
            # Process the data
            df_processed = process_historical_file(df_raw)
            
            print(f"   Processed: {len(df_processed)} securities")
            
            # Check if we already have data for this date
            conn = sqlite3.connect(history_db.db_path)
            existing = pd.read_sql_query(
                "SELECT date FROM daily_snapshots WHERE date = ?", 
                conn, 
                params=(file_date,)
            )
            conn.close()
            
            if not existing.empty:
                print(f"   ⚠️ Data already exists for {file_date}, skipping...")
                duplicate_count += 1
                continue
            
            # Calculate totals
            total_investment = df_processed['Total Cost'].sum() if 'Total Cost' in df_processed.columns else 0
            total_market = df_processed['Market Value'].sum() if 'Market Value' in df_processed.columns else 0
            total_pl = df_processed['Unrealized Gain / (Loss)'].sum() if 'Unrealized Gain / (Loss)' in df_processed.columns else 0
            
            print(f"   Totals - Investment: ${total_investment:,.2f}, Market: ${total_market:,.2f}, P/L: ${total_pl:,.2f}")
            
            # Save to database
            conn = sqlite3.connect(history_db.db_path)
            cursor = conn.cursor()
            
            # Insert daily snapshot
            cursor.execute('''
                INSERT OR REPLACE INTO daily_snapshots 
                (date, total_investment, total_market_value, total_unrealized_pl, cash_balance)
                VALUES (?, ?, ?, ?, ?)
            ''', (file_date, total_investment, total_market, total_pl, 0))
            
            # Insert individual security data
            security_count = 0
            for _, row in df_processed.iterrows():
                security = row.get('Security', 'Unknown')
                quantity = row.get('Quantity', 0)
                market_value = row.get('Market Value', 0)
                total_cost = row.get('Total Cost', 0)
                unrealized_pl = row.get('Unrealized Gain / (Loss)', 0)
                price = market_value / quantity if quantity > 0 else 0
                
                cursor.execute('''
                    INSERT OR REPLACE INTO security_history 
                    (date, security, quantity, market_value, total_cost, unrealized_pl, price)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (file_date, security, quantity, market_value, total_cost, unrealized_pl, price))
                security_count += 1
            
            conn.commit()
            conn.close()
            
            print(f"   ✅ Imported {security_count} securities for {file_date}")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            error_count += 1
    
    # Print summary
    print("\n" + "="*60)
    print("📊 IMPORT SUMMARY")
    print("="*60)
    print(f"✅ Successfully imported: {success_count} files")
    print(f"⚠️ Duplicates skipped: {duplicate_count} files")
    print(f"❌ Errors: {error_count} files")
    print(f"📁 Total processed: {len(excel_files)} files")
    print("="*60)
    
    return success_count, duplicate_count, error_count

def preview_files(folder_path, file_pattern="*.xlsx", n=5):
    """Preview first few files to verify date extraction and data"""
    
    excel_files = glob.glob(os.path.join(folder_path, file_pattern))
    excel_files.sort()
    
    print(f"\n🔍 Preview of first {min(n, len(excel_files))} files:")
    print("-" * 90)
    print(f"{'Filename':<45} {'Date':<12} {'Securities':<10} {'Market Value':<15} {'Investment':<15}")
    print("-" * 90)
    
    for file_path in excel_files[:n]:
        filename = os.path.basename(file_path)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        date = date_match.group(1) if date_match else "NO DATE"
        
        try:
            df_raw = pd.read_excel(file_path, header=None)
            df_sample = process_historical_file(df_raw)
            num_securities = len(df_sample)
            market_value = df_sample['Market Value'].sum() if 'Market Value' in df_sample.columns else 0
            total_cost = df_sample['Total Cost'].sum() if 'Total Cost' in df_sample.columns else 0
            
            print(f"{filename:<45} {date:<12} {num_securities:<10} ${market_value:>12,.2f}  ${total_cost:>12,.2f}")
        except Exception as e:
            print(f"{filename:<45} {date:<12} {'ERROR':<10} {str(e)[:30]}")
    
    print("-" * 90)

def test_single_file(file_path):
    """Test a single file to verify processing works"""
    
    print(f"\n🧪 Testing single file: {os.path.basename(file_path)}")
    print("-" * 60)
    
    try:
        # Read the file
        df_raw = pd.read_excel(file_path, header=None)
        print(f"✅ Read Excel file: {df_raw.shape}")
        
        # Show first few rows of raw data
        print("\n📋 First 3 rows of raw data:")
        print(df_raw.head(3))
        print("\n📋 Last 3 rows of raw data:")
        print(df_raw.tail(3))
        
        # Process the data
        df_processed = process_historical_file(df_raw)
        print(f"\n✅ Processed data: {len(df_processed)} securities")
        
        # Show processed data
        print("\n📋 Processed securities:")
        display_cols = ['Security', 'Quantity', 'Market Value', 'Total Cost', 'Return %']
        available_cols = [col for col in display_cols if col in df_processed.columns]
        print(df_processed[available_cols].to_string())
        
        # Calculate totals
        total_market = df_processed['Market Value'].sum()
        total_cost = df_processed['Total Cost'].sum()
        total_pl = df_processed['Unrealized Gain / (Loss)'].sum()
        
        print(f"\n💰 Totals:")
        print(f"   Market Value: ${total_market:,.2f}")
        print(f"   Total Cost: ${total_cost:,.2f}")
        print(f"   Unrealized P/L: ${total_pl:,.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# MAIN SCRIPT
# ============================================================================

if __name__ == "__main__":
    print("📊 Historical Portfolio Data Importer")
    print("="*60)
    print("For Excel files with format:")
    print('Portfolio (HDF/80464-LI/0 (L.P.D. WELIKALA-200135102901)) - EQUITY')
    print("="*60)
    
    # Ask for folder path
    default_folder = "./historical_data"
    folder = input(f"\nEnter folder path containing renamed Excel files [{default_folder}]: ").strip()
    
    if not folder:
        folder = default_folder
    
    if not os.path.exists(folder):
        print(f"\n❌ Folder not found: {folder}")
        create = input("Create this folder now? (yes/no): ").strip().lower()
        if create in ['yes', 'y']:
            os.makedirs(folder)
            print(f"✅ Created folder: {folder}")
            print(f"Please move your Excel files to: {folder}")
            exit(0)
        else:
            exit(1)
    
    # Count files
    excel_files = glob.glob(os.path.join(folder, "*.xlsx")) + glob.glob(os.path.join(folder, "*.xls"))
    print(f"\n📁 Found {len(excel_files)} Excel files in folder")
    
    if len(excel_files) == 0:
        print("❌ No Excel files found!")
        exit(1)
    
    # Show menu
    print("\n🔧 Options:")
    print("1. Test a single file")
    print("2. Preview all files")
    print("3. Run full import")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == "1":
        # Test a single file
        files = sorted(excel_files)
        print("\nAvailable files:")
        for i, file in enumerate(files[:10]):
            print(f"{i+1}. {os.path.basename(file)}")
        if len(files) > 10:
            print(f"... and {len(files)-10} more")
        
        file_num = input("\nEnter file number to test: ").strip()
        try:
            file_num = int(file_num) - 1
            if 0 <= file_num < len(files):
                test_single_file(files[file_num])
            else:
                print("Invalid file number")
        except:
            print("Invalid input")
    
    elif choice == "2":
        # Preview files
        preview_files(folder)
    
    elif choice == "3":
        # Preview first few files
        preview_files(folder, n=3)
        
        # Confirm before import
        print("\n⚠️ This will import ALL Excel files into your portfolio history database.")
        confirm = input("Do you want to proceed with import? (yes/no): ").strip().lower()
        
        if confirm in ['yes', 'y']:
            # Run import
            success, duplicates, errors = import_historical_files(folder)
            
            print("\n" + "✨"*30)
            print("✨ IMPORT COMPLETE!")
            print("✨"*30)
            
            if success > 0:
                print("\n📊 You can now run your dashboard and see historical data!")
                print("Run: streamlit run dashboard.py")
        else:
            print("Import cancelled.")
    else:
        print("Invalid choice")