# history_db.py - Create this as a separate file first
import sqlite3
import pandas as pd
from datetime import datetime
import os

class PortfolioHistory:
    def __init__(self, db_path='portfolio_history.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create the database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table for daily portfolio snapshots
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                total_investment REAL,
                total_market_value REAL,
                total_unrealized_pl REAL,
                cash_balance REAL DEFAULT 0,
                UNIQUE(date)
            )
        ''')
        
        # Table for individual security history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS security_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                security TEXT NOT NULL,
                quantity INTEGER,
                market_value REAL,
                total_cost REAL,
                unrealized_pl REAL,
                price REAL,
                UNIQUE(date, security)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_snapshot(self, df, cash_balance=0):
        """Save current portfolio state to history"""
        conn = sqlite3.connect(self.db_path)
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Calculate totals
        total_investment = df['Total Cost'].sum() if 'Total Cost' in df.columns else 0
        total_market = df['Market Value'].sum() if 'Market Value' in df.columns else 0
        total_pl = df['Unrealized Gain / (Loss)'].sum() if 'Unrealized Gain / (Loss)' in df.columns else 0
        
        # Insert or replace daily snapshot
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO daily_snapshots 
            (date, total_investment, total_market_value, total_unrealized_pl, cash_balance)
            VALUES (?, ?, ?, ?, ?)
        ''', (today, total_investment, total_market, total_pl, cash_balance))
        
        # Save individual security data
        for _, row in df.iterrows():
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
            ''', (today, security, quantity, market_value, total_cost, unrealized_pl, price))
        
        conn.commit()
        conn.close()
        print(f"✅ Saved snapshot for {today}")
    
    def get_history(self, days=30):
        """Get historical data for the last N days"""
        conn = sqlite3.connect(self.db_path)
        
        query = f'''
            SELECT * FROM daily_snapshots 
            ORDER BY date DESC 
            LIMIT {days}
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert date to datetime and sort
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        
        return df
    
    def get_security_history(self, security, days=30):
        """Get historical data for a specific security"""
        conn = sqlite3.connect(self.db_path)
        
        query = f'''
            SELECT * FROM security_history 
            WHERE security = ?
            ORDER BY date DESC 
            LIMIT {days}
        '''
        df = pd.read_sql_query(query, conn, params=(security,))
        conn.close()
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        
        return df
    
    def get_all_history(self):
        """Get all historical data"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM daily_snapshots ORDER BY date", conn)
        conn.close()
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        
        return df

# Test the database
if __name__ == "__main__":
    # Test with sample data
    history = PortfolioHistory()
    print("✅ Database initialized")

# Add these functions to history_db.py

def export_history(self):
    """Export all historical data to CSV"""
    conn = sqlite3.connect(self.db_path)
    
    # Export daily snapshots
    daily_df = pd.read_sql_query("SELECT * FROM daily_snapshots ORDER BY date", conn)
    
    # Export security history
    security_df = pd.read_sql_query("SELECT * FROM security_history ORDER BY date, security", conn)
    
    conn.close()
    
    return daily_df, security_df

def import_history(self, daily_df, security_df):
    """Import historical data from CSV"""
    conn = sqlite3.connect(self.db_path)
    
    # Clear existing data
    cursor = conn.cursor()
    cursor.execute("DELETE FROM daily_snapshots")
    cursor.execute("DELETE FROM security_history")
    
    # Import new data
    daily_df.to_sql('daily_snapshots', conn, if_exists='append', index=False)
    security_df.to_sql('security_history', conn, if_exists='append', index=False)
    
    conn.commit()
    conn.close()