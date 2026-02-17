# dashboard.py - Complete with Historical Tracking
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import re
import chardet
import sqlite3
import os

st.set_page_config(page_title="Portfolio Dashboard", page_icon="📊", layout="wide")

st.title("📊 Investment Portfolio Dashboard")

# Initialize session state for history database
if 'history_db_initialized' not in st.session_state:
    st.session_state.history_db_initialized = True
    st.session_state.days_to_show = 30
    st.session_state.cash_balance = 0.0

# ============================================================================
# HISTORY DATABASE CLASS
# ============================================================================

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
        return True
    
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
    
    def export_history(self):
        """Export all historical data to CSV"""
        conn = sqlite3.connect(self.db_path)
        
        # Export daily snapshots
        daily_df = pd.read_sql_query("SELECT * FROM daily_snapshots ORDER BY date", conn)
        
        # Export security history
        security_df = pd.read_sql_query("SELECT * FROM security_history ORDER BY date, security", conn)
        
        conn.close()
        
        return daily_df, security_df
    
    def delete_snapshot(self, date):
        """Delete a specific snapshot by date"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM daily_snapshots WHERE date = ?", (date,))
        cursor.execute("DELETE FROM security_history WHERE date = ?", (date,))
        conn.commit()
        conn.close()

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def detect_encoding(file_content):
    """Detect the encoding of a file"""
    result = chardet.detect(file_content)
    return result['encoding']

def clean_numeric(value):
    """Convert string numbers with commas to float"""
    if pd.isna(value):
        return 0.0
    value = str(value)
    # Remove commas, spaces, and any other non-numeric characters except decimal and minus
    value = re.sub(r'[^\d.-]', '', value)
    try:
        return float(value) if value else 0.0
    except:
        return 0.0

def process_portfolio_data(df):
    """Clean and prepare the portfolio data from Atrad CSV"""
    
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
    
    # Ensure we have the right number of columns
    if len(df_clean.columns) >= len(columns):
        df_clean.columns = columns[:len(df_clean.columns)]
    else:
        # If fewer columns, use generic names
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
    
    # Convert Quantity to integer (if it exists)
    if 'Quantity' in df_clean.columns:
        df_clean['Quantity'] = df_clean['Quantity'].fillna(0).astype(int)
    
    # Calculate additional metrics (with safety checks)
    if all(col in df_clean.columns for col in ['Total Cost', 'Quantity', 'Market Value']):
        # Avoid division by zero
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
    
    if all(col in df_clean.columns for col in ['Current Price', 'BES Price']):
        df_clean['Current vs BES %'] = df_clean.apply(
            lambda row: ((row['Current Price'] / row['BES Price']) - 1) * 100 if row['BES Price'] != 0 else 0,
            axis=1
        )
    
    if 'Market Value' in df_clean.columns:
        total_market = df_clean['Market Value'].sum()
        df_clean['Position Weight'] = df_clean.apply(
            lambda row: (row['Market Value'] / total_market * 100) if total_market != 0 else 0,
            axis=1
        )
    
    return df_clean

# ============================================================================
# CHART FUNCTIONS
# ============================================================================

def create_time_series_charts(history_df):
    """Create time series charts from historical data"""
    
    if history_df.empty:
        st.info("No historical data yet. Use the sidebar to save your first snapshot!")
        return
    
    # Create two rows of charts
    
    # Line chart for Investment vs Market Value
        # Summary metrics
    latest = history_df.iloc[-1]
    first = history_df.iloc[0]
        
    total_return = latest['total_market_value'] - latest['total_investment']
    total_return_pct = (total_return / latest['total_investment'] * 100) if latest['total_investment'] != 0 else 0
        
    period_return = latest['total_market_value'] - first['total_market_value']
    period_return_pct = (period_return / first['total_market_value'] * 100) if first['total_market_value'] != 0 else 0
        
    st.metric("Current Value", f"Rs.{latest['total_market_value']:,.2f}")
    st.metric("Total Return", f"Rs.{total_return:,.2f}", f"{total_return_pct:.1f}%")
    st.metric("Period Return", f"Rs.{period_return:,.2f}", f"{period_return_pct:.1f}%")
    st.metric("Days Tracked", f"{len(history_df)} days")


    fig = go.Figure()
        
    fig.add_trace(go.Scatter(
            x=history_df['date'],
            y=history_df['total_investment'],
            mode='lines+markers',
            name='Total Investment',
            line=dict(color='blue', width=2),
            hovertemplate='Date: %{x|%Y-%m-%d}<br>Investment: Rs.%{y:,.2f}<extra></extra>'
        ))
        
    fig.add_trace(go.Scatter(
            x=history_df['date'],
            y=history_df['total_market_value'],
            mode='lines+markers',
            name='Market Value',
            line=dict(color='green', width=2),
            hovertemplate='Date: %{x|%Y-%m-%d}<br>Market Value: Rs.%{y:,.2f}<extra></extra>'
        ))
        
        # Add filled area between lines to show profit/loss
    fig.add_trace(go.Scatter(
            x=history_df['date'],
            y=history_df['total_market_value'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
    fig.add_trace(go.Scatter(
            x=history_df['date'],
            y=history_df['total_investment'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0,255,0,0.1)' if history_df['total_market_value'].iloc[-1] > history_df['total_investment'].iloc[-1] else 'rgba(255,0,0,0.1)',
            showlegend=False,
            hoverinfo='skip'
        ))
        
    fig.update_layout(
            title='📈 Portfolio Value Over Time',
            xaxis_title='Date',
            yaxis_title='Value (Rs.)',
            hovermode='x unified',
            height=400
        )
        
    st.plotly_chart(fig, use_container_width=True)
    
    
    # Second row - Gains chart
    fig_gains = go.Figure()
    
    # Calculate cumulative gains
    history_df['cumulative_gains'] = history_df['total_market_value'] - history_df['total_investment']
    
    fig_gains.add_trace(go.Scatter(
        x=history_df['date'],
        y=history_df['cumulative_gains'],
        mode='lines+markers',
        name='Unrealized P/L',
        line=dict(color='orange', width=3),
        fill='tozeroy',
        fillcolor='rgba(255,165,0,0.1)',
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Gain/Loss: Rs.%{y:,.2f}<extra></extra>'
    ))
    
    # Add horizontal line at zero
    fig_gains.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig_gains.update_layout(
        title='💰 Cumulative Gains/Losses Over Time',
        xaxis_title='Date',
        yaxis_title='Gain/Loss (Rs.)',
        hovermode='x unified',
        height=300
    )
    
    st.plotly_chart(fig_gains, use_container_width=True)

def display_security_history(df, history_db):
    """Display historical data for individual securities"""
    
    st.subheader("📊 Individual Security History")
    
    # Select security to view
    securities = df['Security'].tolist() if 'Security' in df.columns else []
    
    if securities:
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_security = st.selectbox("Select Security:", securities, key="sec_history")
        with col2:
            sec_days = st.number_input("Days of History", min_value=7, max_value=365, value=30, key="sec_days")
        
        if selected_security:
            sec_history = history_db.get_security_history(selected_security, days=sec_days)
            
            if not sec_history.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Price history
                    fig_price = px.line(
                        sec_history, 
                        x='date', 
                        y='price',
                        title=f'{selected_security} - Price History',
                        markers=True
                    )
                    fig_price.update_layout(height=300)
                    st.plotly_chart(fig_price, use_container_width=True)
                
                with col2:
                    # P/L history
                    fig_pl = px.line(
                        sec_history, 
                        x='date', 
                        y='unrealized_pl',
                        title=f'{selected_security} - Gain/Loss History',
                        markers=True
                    )
                    fig_pl.update_layout(height=300)
                    st.plotly_chart(fig_pl, use_container_width=True)
                
                # Show data table
                with st.expander("View Historical Data"):
                    st.dataframe(sec_history, use_container_width=True)
            else:
                st.info(f"No history yet for {selected_security}")
    else:
        st.info("No securities found in current portfolio")

def manage_history(history_db):
    """Allow users to manage their historical data"""
    st.subheader("📚 History Management")
    
    # Get all history
    all_history = history_db.get_all_history()
    
    if not all_history.empty:
        st.write(f"Total snapshots: {len(all_history)}")
        
        # Show history table
        st.dataframe(
            all_history[['date', 'total_investment', 'total_market_value', 'total_unrealized_pl']]
            .style.format({
                'total_investment': '${:,.2f}',
                'total_market_value': '${:,.2f}',
                'total_unrealized_pl': '${:,.2f}'
            }),
            use_container_width=True
        )
        
        # Delete option
        st.warning("⚠️ Delete Snapshot")
        dates_to_delete = st.multiselect(
            "Select dates to delete:",
            options=all_history['date'].dt.strftime('%Y-%m-%d').tolist()
        )
        
        if dates_to_delete and st.button("Delete Selected Snapshots", type="primary"):
            for date in dates_to_delete:
                history_db.delete_snapshot(date)
            st.success(f"Deleted {len(dates_to_delete)} snapshots")
            st.rerun()
    else:
        st.info("No historical data yet")

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

# Sidebar
with st.sidebar:
    st.header("📁 Portfolio Data")
    uploaded_file = st.file_uploader(
        "Upload Atrad CSV",
        type=['csv', 'xlsx'],
        help="Download your portfolio from Atrad trading platform"
    )
    
    if uploaded_file:
        st.success(f"Loaded: {uploaded_file.name}")
    
    st.divider()
    
    # Historical Tracking Section
    st.header("Historical Tracking")
    
    # Initialize history database
    history_db = PortfolioHistory()
    
    # Days to show slider
    days_to_show = st.slider("Days of History to Show", 7, 90, 30)
    st.session_state.days_to_show = days_to_show
    
    # Cash balance input
    cash_balance = st.number_input("Cash Balance ($)", min_value=0.0, value=0.0, step=1000.0)
    st.session_state.cash_balance = cash_balance
    
    # Save snapshot button (will be used after data is processed)
    if uploaded_file and st.button("Save Today's Snapshot"):
        st.session_state.save_snapshot = True
    
    st.divider()
    
    # History Management
    with st.expander("Manage History"):
        if st.button("View/Manage History"):
            st.session_state.show_history_manager = True
        
        # Export option
        if st.button("Export History"):
            daily_df, security_df = history_db.export_history()
            
            # Create download buttons
            csv1 = daily_df.to_csv(index=False)
            csv2 = security_df.to_csv(index=False)
            
            st.download_button(
                "Download Daily Snapshots",
                data=csv1,
                file_name=f"daily_snapshots_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            st.download_button(
                "Download Security History",
                data=csv2,
                file_name=f"security_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    st.divider()
    st.caption(f"Session: {datetime.now().strftime('%H:%M:%S')}")

# Main content
if uploaded_file is not None:
    try:
        # Read file content
        file_content = uploaded_file.getvalue()
        
        # Try different encodings
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
        
        df_raw = None
        used_encoding = None
        
        # First, try to detect encoding
        detected_encoding = detect_encoding(file_content)
        if detected_encoding:
            encodings_to_try.insert(0, detected_encoding)
        
        # Try each encoding
        for encoding in encodings_to_try:
            try:
                # Reset file pointer
                uploaded_file.seek(0)
                
                # Check file type
                if uploaded_file.name.endswith('.xlsx'):
                    df_raw = pd.read_excel(uploaded_file, header=None)
                else:
                    df_raw = pd.read_csv(uploaded_file, header=None, encoding=encoding)
                
                used_encoding = encoding
                st.sidebar.success(f"✅ Used encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                continue
        
        if df_raw is None:
            # Last resort: try with error handling
            uploaded_file.seek(0)
            if uploaded_file.name.endswith('.xlsx'):
                df_raw = pd.read_excel(uploaded_file, header=None)
            else:
                df_raw = pd.read_csv(uploaded_file, header=None, encoding='utf-8', errors='ignore')
            used_encoding = 'utf-8 (with errors ignored)'
            st.sidebar.warning("⚠️ Used fallback encoding")
        
        # Show raw data structure (for debugging)
        with st.expander("📋 Raw Data Structure", expanded=False):
            st.write(f"**File info:**")
            st.write(f"- Encoding used: {used_encoding}")
            st.write(f"- Raw shape: {df_raw.shape}")
            st.write(f"- First 5 rows:")
            st.dataframe(df_raw.head(), use_container_width=True)
        
        # Process data
        df = process_portfolio_data(df_raw)
        
        
        # Save snapshot if requested
        if st.session_state.get('save_snapshot', False):
            with st.spinner("Saving snapshot..."):
                history_db.save_snapshot(df, st.session_state.cash_balance)
                st.session_state.save_snapshot = False
                st.success("✅ Portfolio snapshot saved to history!")
        
        # Get historical data
        history_df = history_db.get_history(st.session_state.days_to_show)
        
        # ================================================================
        # HISTORICAL CHARTS SECTION (TOP)
        # ================================================================
        st.subheader("📈 Portfolio Performance Over Time")
        create_time_series_charts(history_df)
        
        # ================================================================
        # CURRENT PORTFOLIO SECTION
        # ================================================================
        
        # Display KPIs
        st.subheader("📊 Current Portfolio Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_cost = df['Total Cost'].sum() if 'Total Cost' in df.columns else 0
            st.metric("💰 Total Investment", f"${total_cost:,.2f}")
        
        with col2:
            total_market = df['Market Value'].sum() if 'Market Value' in df.columns else 0
            delta = total_market - total_cost if total_cost != 0 else 0
            st.metric("📈 Market Value", f"${total_market:,.2f}",
                     delta=f"${delta:,.2f}")
        
        with col3:
            total_pl = df['Unrealized Gain / (Loss)'].sum() if 'Unrealized Gain / (Loss)' in df.columns else 0
            pl_pct = (total_pl / total_cost * 100) if total_cost != 0 else 0
            st.metric("💵 Unrealized P/L", f"${total_pl:,.2f}",
                     delta=f"{pl_pct:.1f}%")
        
        with col4:
            if 'Unrealized Gain / (Loss)' in df.columns:
                winners = len(df[df['Unrealized Gain / (Loss)'] > 0])
                losers = len(df[df['Unrealized Gain / (Loss)'] < 0])
                total = len(df)
                win_rate = (winners/total*100) if total > 0 else 0
                st.metric("📊 Win Rate", f"{win_rate:.1f}%",
                         delta=f"{winners}W / {losers}L")
            else:
                st.metric("📊 Holdings", f"{len(df)} Securities")
        
        # Portfolio Table
        st.subheader("📋 Current Holdings")
        
        # Search filter
        search = st.text_input("🔍 Search securities:", placeholder="Type to filter...")
        
        # Filter data
        if search and 'Security' in df.columns:
            filtered_df = df[df['Security'].str.contains(search, case=False, na=False)]
        else:
            filtered_df = df
        
        # Select columns to display
        available_cols = filtered_df.columns.tolist()
        display_cols = []
        
        preferred_cols = [
            'Security', 'Quantity', 'Market Value', 'Total Cost',
            'Unrealized Gain / (Loss)', 'Return %', 'Current vs BES %',
            'Position Weight'
        ]
        
        for col in preferred_cols:
            if col in available_cols:
                display_cols.append(col)
        
        if display_cols:
            # Create formatted dataframe
            display_df = filtered_df[display_cols].copy()
            
            # Format the dataframe
            styled_df = display_df.style.format({
                'Market Value': '${:,.2f}',
                'Total Cost': '${:,.2f}',
                'Unrealized Gain / (Loss)': '${:,.2f}',
                'Return %': '{:.2f}%',
                'Current vs BES %': '{:.2f}%',
                'Position Weight': '{:.2f}%',
                'Quantity': '{:,.0f}'
            })
            
            st.dataframe(styled_df, use_container_width=True, height=400)
        else:
            st.dataframe(filtered_df, use_container_width=True, height=400)
        
        # Simple Charts
        if len(filtered_df) > 0 and 'Market Value' in filtered_df.columns:
            st.subheader("📈 Visual Analytics")
            
            tab1, tab2 = st.tabs(["📊 Allocation", "📈 Performance"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Portfolio Allocation Pie Chart
                    fig_pie = px.pie(
                        filtered_df, 
                        values='Market Value', 
                        names='Security' if 'Security' in filtered_df.columns else filtered_df.index,
                        title='Portfolio Allocation by Market Value'
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    if 'Position Weight' in filtered_df.columns:
                        # Position Weights Bar Chart
                        plot_df = filtered_df.copy()
                        if 'Security' in plot_df.columns:
                            plot_df = plot_df.sort_values('Position Weight', ascending=True)
                            fig_weights = px.bar(
                                plot_df,
                                x='Position Weight',
                                y='Security',
                                orientation='h',
                                title='Position Weights (%)',
                                color='Return %' if 'Return %' in plot_df.columns else None,
                                color_continuous_scale=['red', 'yellow', 'green']
                            )
                            fig_weights.update_layout(xaxis_title="Weight (%)")
                            st.plotly_chart(fig_weights, use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'Return %' in filtered_df.columns and 'Security' in filtered_df.columns:
                        # Returns Bar Chart
                        plot_df = filtered_df.sort_values('Return %', ascending=True)
                        fig_returns = px.bar(
                            plot_df,
                            x='Return %',
                            y='Security',
                            orientation='h',
                            title='Returns by Security',
                            color='Return %',
                            color_continuous_scale=['red', 'yellow', 'green']
                        )
                        fig_returns.update_layout(xaxis_title="Return (%)")
                        st.plotly_chart(fig_returns, use_container_width=True)
                
                with col2:
                    if 'Unrealized Gain / (Loss)' in filtered_df.columns and 'Security' in filtered_df.columns:
                        # Gain/Loss Bar Chart
                        plot_df = filtered_df.sort_values('Unrealized Gain / (Loss)', ascending=True)
                        fig_pl = px.bar(
                            plot_df,
                            x='Unrealized Gain / (Loss)',
                            y='Security',
                            orientation='h',
                            title='Gain/Loss by Security ($)',
                            color='Unrealized Gain / (Loss)',
                            color_continuous_scale=['red', 'yellow', 'green']
                        )
                        fig_pl.update_layout(xaxis_title="Gain/Loss ($)")
                        st.plotly_chart(fig_pl, use_container_width=True)
        
        # Summary Statistics
        st.subheader("📊 Portfolio Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📈 Top Performers**")
            if 'Return %' in df.columns and 'Security' in df.columns:
                top_3 = df.nlargest(3, 'Return %')[['Security', 'Return %']]
                for _, row in top_3.iterrows():
                    st.write(f"• {row['Security']}: {row['Return %']:.1f}%")
            else:
                st.write("No performance data")
        
        with col2:
            st.markdown("**📉 Bottom Performers**")
            if 'Return %' in df.columns and 'Security' in df.columns:
                bottom_3 = df.nsmallest(3, 'Return %')[['Security', 'Return %']]
                for _, row in bottom_3.iterrows():
                    st.write(f"• {row['Security']}: {row['Return %']:.1f}%")
            else:
                st.write("No performance data")
        
        with col3:
            st.markdown("**📊 Portfolio Stats**")
            st.write(f"• Total Securities: {len(df)}")
            if 'Return %' in df.columns:
                st.write(f"• Avg Return: {df['Return %'].mean():.1f}%")
                st.write(f"• Best Return: {df['Return %'].max():.1f}%")
                st.write(f"• Worst Return: {df['Return %'].min():.1f}%")
        
        # ================================================================
        # SECURITY HISTORY SECTION
        # ================================================================
        with st.expander("📈 View Individual Security History"):
            display_security_history(df, history_db)
        
        # ================================================================
        # HISTORY MANAGER SECTION
        # ================================================================
        if st.session_state.get('show_history_manager', False):
            with st.expander("📚 History Manager", expanded=True):
                manage_history(history_db)
                if st.button("Close History Manager"):
                    st.session_state.show_history_manager = False
                    st.rerun()
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Cleaned Data",
            data=csv,
            file_name=f"cleaned_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)
        
        # Troubleshooting section
        st.subheader("🔍 Troubleshooting")
        st.write("Please check that your file is a valid CSV or Excel file from Atrad.")

else:
    # Welcome screen
    st.info("👈 Please upload your Atrad CSV file to view your portfolio dashboard")
    
    # Show instructions
    st.subheader("📋 How to use:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Dashboard Features:
        - **Real-time portfolio tracking**
        - **Interactive charts and visualizations**
        - **Profit/loss analysis**
        - **Search and filter securities**
        - **Download cleaned data**
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Historical Tracking:
        1. Upload your CSV file
        2. Enter your cash balance
        3. Click "Save Today's Snapshot"
        4. Watch your history grow!
        
        **Supported formats:** CSV, Excel (.xlsx)
        """)

# Footer
st.divider()
st.caption(f"Portfolio Dashboard with Historical Tracking | Built for Atrad | Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")