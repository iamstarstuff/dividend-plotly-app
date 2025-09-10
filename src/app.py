"""
Enhanced Dividend Analysis App with 4 Dropdown Filters
1. Stock Symbol + Company + Exchange
2. Stock Exchange Filter
3. 10-Year Average Yield Range
4. Timeframe (3, 5, 10 years)
"""
import dash
from dash import dcc, html, Input, Output, State, callback
import dash.dependencies
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
import os
import sys
import yfinance as yf

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data import DataLoader
from update_scripts.index_dividend_fetcher import IndexBasedDividendFetcher, check_data_freshness, main as update_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Custom CSS for dropdown styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .custom-dropdown .Select-value-label,
            .custom-dropdown .Select-placeholder,
            .custom-dropdown .Select-input > input,
            .custom-dropdown .VirtualizedSelectOption {
                color: #fff !important;
            }
            .custom-dropdown .Select-menu-outer {
                background-color: #333 !important;
            }
            .custom-dropdown .VirtualizedSelectFocusedOption {
                background-color: #555 !important;
                color: #fff !important;
            }
            
            /* Custom Input Styling */
            .custom-input {
                transition: all 0.3s ease !important;
            }
            .custom-input:hover {
                border-color: #667eea !important;
                box-shadow: 0 4px 8px rgba(102, 126, 234, 0.15) !important;
                transform: translateY(-1px) !important;
            }
            .custom-input:focus {
                border-color: #667eea !important;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25), 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
                transform: translateY(-1px) !important;
                outline: none !important;
            }
            .custom-input::placeholder {
                color: #a0aec0 !important;
                font-style: italic !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Initialize data loader and load data once at startup
data_loader = DataLoader()

# Load data once at startup
def load_dividend_data():
    """Unified function to load dividend data with automatic updates"""
    try:
        # Check for HIGH-YIELD data first (our new discovery)
        high_yield_csv = os.path.join(os.path.dirname(__file__), 'data', 'focused_high_yield_dividends_20250906_215020.csv')
        
        # Check for index-based data file (fallback)
        index_csv = os.path.join(os.path.dirname(__file__), 'data', 'index_dividend_data.csv')
        
        # Always load both datasets and merge for complete coverage
        datasets_to_merge = []
        
        if os.path.exists(high_yield_csv):
            logger.info("✅ Loading HIGH-YIELD dividend data (global discovery)")
            high_yield_df = pd.read_csv(high_yield_csv)
            high_yield_df['data_source'] = 'high_yield_discovery'
            datasets_to_merge.append(high_yield_df)
            
        if os.path.exists(index_csv):
            logger.info("✅ Loading INDEX-based dividend data (comprehensive coverage)")
            index_df = pd.read_csv(index_csv)
            index_df['data_source'] = 'index_based'
            datasets_to_merge.append(index_df)
        
        if datasets_to_merge:
            # Combine all available datasets
            combined_df = pd.concat(datasets_to_merge, ignore_index=True)
            
            # Remove duplicates, keeping high-yield version first
            combined_df = combined_df.drop_duplicates(
                subset=['ticker_symbol', 'dividend_date'], 
                keep='first'
            )
            
            logger.info(f"📊 Merged dividend database: {len(combined_df)} records from {len(combined_df['ticker_symbol'].unique())} companies")
            
            # Count data sources
            high_yield_count = len(combined_df[combined_df.get('data_source', '') == 'high_yield_discovery']['ticker_symbol'].unique())
            index_count = len(combined_df[combined_df.get('data_source', '') == 'index_based']['ticker_symbol'].unique())
            logger.info(f"🔥 {high_yield_count} high-yield discoveries + 📊 {index_count} index companies")
            
            return combined_df
        
        logger.warning("⚠️ No dividend data files found - using fallback data loader")
        
        # Check if data needs updating and update if necessary
        if not check_data_freshness(index_csv):
            logger.info("🔄 Data needs updating - fetching fresh data...")
            try:
                update_data()
            except Exception as e:
                logger.warning(f"⚠️ Auto-update failed: {e}, using existing data")
        
        # Load the data (priority: index_data -> exchange_data -> data_loader)
        if os.path.exists(index_csv):
            return pd.read_csv(index_csv)
        
        # Fallback to old exchange data
        exchange_csv = os.path.join(os.path.dirname(__file__), 'data', 'exchange_dividend_data.csv')
        if os.path.exists(exchange_csv):
            return pd.read_csv(exchange_csv)
        
        # Last resort: data loader
        return data_loader.load_data()
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        # Ultimate fallback
        try:
            return data_loader.load_data()
        except:
            return pd.DataFrame()

# Load data once at startup - cache it globally
GLOBAL_DIVIDEND_DATA = load_dividend_data()
logger.info(f"🚀 Startup: Loaded {len(GLOBAL_DIVIDEND_DATA)} records from {len(GLOBAL_DIVIDEND_DATA['ticker_symbol'].unique()) if not GLOBAL_DIVIDEND_DATA.empty else 0} companies")

# Global storage for dynamically fetched stock data
DYNAMIC_STOCK_DATA = {}
# Track the last fetched symbol for auto-selection
LAST_FETCHED_SYMBOL = None

# ====================================================================================================
# Global storage for dynamically fetched stock data
DYNAMIC_STOCK_DATA = {}
# Track the last fetched symbol for auto-selection
LAST_FETCHED_SYMBOL = None

def get_cached_data():
    """Get cached dividend data (no repeated loading)"""
    return GLOBAL_DIVIDEND_DATA

def get_cached_data_with_dynamic():
    """Get cached data including any dynamically fetched stocks"""
    global DYNAMIC_STOCK_DATA
    
    # Get the base cached data
    base_data = get_cached_data()
    
    # If no dynamic data, return base data
    if not DYNAMIC_STOCK_DATA:
        return base_data
    
    # Combine dynamic data with base data
    dynamic_dfs = []
    for symbol, data in DYNAMIC_STOCK_DATA.items():
        dynamic_dfs.append(data)
    
    if dynamic_dfs:
        combined_dynamic = pd.concat(dynamic_dfs, ignore_index=True)
        # Combine with base data
        combined_data = pd.concat([combined_dynamic, base_data], ignore_index=True)
        return combined_data
    
    return base_data

def get_enhanced_stock_options():
    """Get stock options from index-based data with automatic daily updates"""
    try:
        df = get_cached_data_with_dynamic()
        
        if df.empty:
            return [{'label': 'No Data Available', 'value': 'none'}]
        
        # Create comprehensive exchange-to-country mapping
        exchange_country_map = {
            # Index data mappings
            'NASDAQ': 'USA', 'NYSE': 'USA', 'NSE': 'India', 'TSX': 'Canada',
            # High-yield data mappings  
            'ASE': 'USA', 'NGM': 'USA', 'NMS': 'USA', 'NYQ': 'USA',  # NYQ is NASDAQ Global Select (US)
            'PCX': 'Pacific Exchange (USA)', 'TOR': 'Canada',
            # Indian Stock Exchanges
            'NSI': 'India',  # National Stock Exchange of India
            'BSE': 'India',  # Bombay Stock Exchange
            # Additional common exchanges
            'LSE': 'UK', 'ASX': 'Australia', 'JPX': 'Japan', 'HKG': 'Hong Kong',
            'SWX': 'Switzerland', 'FRA': 'Germany', 'AMS': 'Netherlands'
        }
        
        # Process the dataframe
        df['dividend_date'] = pd.to_datetime(df['dividend_date'])
        df['dividend_per_share'] = pd.to_numeric(df['dividend_per_share'], errors='coerce')
        df['dividend_yield_pct'] = pd.to_numeric(df['dividend_yield_pct'], errors='coerce')
        df = df.dropna(subset=['dividend_per_share', 'dividend_yield_pct'])
        
        # Add country mapping if not present or incorrect
        if 'country' not in df.columns:
            df['country'] = df['exchange'].map(exchange_country_map).fillna('Unknown')
        else:
            # Update any missing or incorrect country mappings
            df['country'] = df.apply(
                lambda row: exchange_country_map.get(row['exchange'], row.get('country', 'Unknown')), 
                axis=1
            )
        
        # Calculate average yield per stock with enhanced info
        stock_stats = df.groupby(['ticker_symbol', 'company_name', 'exchange', 'country']).agg({
            'dividend_yield_pct': 'mean',
            'dividend_date': 'count',
            'data_source': 'first'  # Track data source
        }).round(2)
        
        stock_stats.columns = ['avg_yield', 'dividend_count', 'data_source']
        stock_stats = stock_stats.reset_index()
        
        # Add high-yield indicator based on data source
        stock_stats['is_high_yield'] = stock_stats['data_source'] == 'high_yield_discovery'
        
        # Filter for stocks with at least 3 dividend records
        stock_stats = stock_stats[stock_stats['dividend_count'] >= 3]
        
        # Sort by high-yield first, then by exchange, then by average yield descending
        stock_stats = stock_stats.sort_values(['is_high_yield', 'exchange', 'avg_yield'], ascending=[False, True, False])
        
        # Use all qualified stocks instead of limiting to top 50 per exchange
        top_stocks_df = stock_stats
        
        # Create dropdown options grouped by exchange with country names
        options = []
        current_exchange = None
        
        for _, row in top_stocks_df.iterrows():
            # Add exchange headers with country names
            if row['exchange'] != current_exchange:
                if current_exchange is not None:  # Add separator
                    options.append({'label': '─' * 60, 'value': 'separator', 'disabled': True})
                
                # Enhanced header with country and flag emoji
                country_flags = {
                    'USA': '🇺🇸', 'India': '🇮🇳', 'Canada': '🇨🇦', 'UK': '🇬🇧', 
                    'Australia': '🇦🇺', 'Japan': '🇯🇵', 'Hong Kong': '🇭🇰',
                    'South Korea': '🇰🇷', 'Switzerland': '🇨🇭', 'Germany': '🇩🇪', 
                    'Netherlands': '🇳🇱', 'Pacific Exchange (USA)': '🇺🇸'
                }
                flag = country_flags.get(row['country'], '🌍')
                
                options.append({
                    'label': f"{flag} {row['exchange']} - {row['country']}",
                    'value': f"header_{row['exchange']}",
                    'disabled': True
                })
                current_exchange = row['exchange']
            
            # Add stock option with high-yield indicator
            high_yield_indicator = "🔥 " if row.get('is_high_yield', False) else ""
            yield_badge = "🚀" if row['avg_yield'] >= 10 else "⭐" if row['avg_yield'] >= 5 else ""
            
            label = f"{high_yield_indicator}{yield_badge}{row['ticker_symbol']} - {row['company_name'][:35]}{'...' if len(row['company_name']) > 35 else ''} (Avg: {row['avg_yield']:.1f}%)"
            options.append({
                'label': label,
                'value': row['ticker_symbol']
            })
        
        # Count high-yield stocks and log summary
        high_yield_count = len([opt for opt in options if '🔥' in opt.get('label', '')])
        total_countries = len(top_stocks_df['country'].unique())
        logger.info(f"📊 Created dropdown: {len([opt for opt in options if not opt.get('disabled')])} stocks across {top_stocks_df['exchange'].nunique()} exchanges in {total_countries} countries (🔥 {high_yield_count} high-yield discoveries)")
        return options
    except Exception as e:
        logger.error(f"Error getting stock options: {e}")
        return [{'label': 'Error Loading Data', 'value': 'none'}]

def get_stock_exchanges():
    """Get unique stock exchanges"""
    try:
        df = get_cached_data_with_dynamic()
        
        if df.empty:
            return ['NYSE', 'NASDAQ', 'LSE', 'TSX', 'ASX', 'NSE']  # Default exchanges
        
        exchanges = sorted(df['exchange'].unique())
        return exchanges
    except Exception as e:
        logger.error(f"Error getting exchanges: {e}")
        return ['NYSE', 'NASDAQ', 'LSE', 'TSX', 'ASX', 'NSE']

def get_enhanced_exchange_options():
    """Get exchange options for dropdown with country information"""
    try:
        df = get_cached_data_with_dynamic()
        
        if df.empty:
            return [{'label': 'All Exchanges', 'value': 'ALL'}]
        
        # Create comprehensive exchange-to-country mapping
        exchange_country_map = {
            # Index data mappings
            'NASDAQ': 'USA', 'NYSE': 'USA', 'NSE': 'India', 'TSX': 'Canada',
            # High-yield data mappings  
            'ASE': 'USA', 'NGM': 'USA', 'NMS': 'USA', 'NYQ': 'USA',  # NYQ is NASDAQ Global Select (US)
            'PCX': 'Pacific Exchange (USA)', 'TOR': 'Canada',
            # Indian Stock Exchanges
            'NSI': 'India',  # National Stock Exchange of India
            'BSE': 'India',  # Bombay Stock Exchange
            # Additional common exchanges
            'LSE': 'UK', 'ASX': 'Australia', 'JPX': 'Japan', 'HKG': 'Hong Kong',
            'SWX': 'Switzerland', 'FRA': 'Germany', 'AMS': 'Netherlands'
        }
        
        # Country flag mapping
        country_flags = {
            'USA': '🇺🇸', 'India': '🇮🇳', 'Canada': '🇨🇦', 'UK': '🇬🇧', 
            'Australia': '🇦🇺', 'Japan': '🇯🇵', 'Hong Kong': '🇭🇰',
            'South Korea': '🇰🇷', 'Switzerland': '🇨🇭', 'Germany': '🇩🇪', 
            'Netherlands': '🇳🇱', 'Pacific Exchange (USA)': '🇺🇸'
        }
        
        # Get exchanges from data and add country info
        exchanges = sorted(df['exchange'].unique())
        options = [{'label': '🌍 All Exchanges', 'value': 'ALL'}]
        
        for exchange in exchanges:
            country = exchange_country_map.get(exchange, 'Unknown')
            flag = country_flags.get(country, '🌍')
            label = f"{flag} {exchange} - {country}"
            options.append({'label': label, 'value': exchange})
        
        return options
    except Exception as e:
        logger.error(f"Error getting exchange options: {e}")
        return [{'label': '🌍 All Exchanges', 'value': 'ALL'}]

# Get initial data for dropdowns
stock_options = get_enhanced_stock_options()
exchange_options = get_enhanced_exchange_options()

def fetch_stock_dividend_data(symbol: str) -> Dict:
    """
    Fetch dividend data for a given stock symbol using yfinance
    Returns a dictionary with status and data/error information
    """
    try:
        # Clean and format the symbol
        symbol = symbol.strip().upper()
        
        # Create yfinance ticker object
        ticker = yf.Ticker(symbol)
        
        # Get basic info first to validate the symbol
        info = ticker.info
        if not info or 'symbol' not in info:
            return {
                'success': False,
                'error': f"Stock symbol '{symbol}' not found. Please check the symbol and try again."
            }
        
        # Get dividend data
        dividends = ticker.dividends
        if dividends.empty:
            return {
                'success': False,
                'error': f"No dividend data found for '{symbol}'. This stock may not pay dividends."
            }
        
        # Get recent stock price
        hist = ticker.history(period="5d")
        current_price = hist['Close'].iloc[-1] if not hist.empty else None
        
        # Process dividend data
        dividend_df = dividends.reset_index()
        dividend_df.columns = ['dividend_date', 'dividend_per_share']
        
        # Calculate yield for each dividend payment
        dividend_yields = []
        for _, row in dividend_df.iterrows():
            div_date = row['dividend_date']
            div_amount = row['dividend_per_share']
            
            # Get stock price around dividend date
            try:
                # Convert timezone-aware date to date object for yfinance
                if hasattr(div_date, 'tz') and div_date.tz is not None:
                    div_date_naive = div_date.tz_localize(None)
                else:
                    div_date_naive = div_date
                    
                price_hist = ticker.history(start=div_date_naive - timedelta(days=5), 
                                          end=div_date_naive + timedelta(days=5))
                if not price_hist.empty:
                    price_on_date = price_hist['Close'].iloc[0]
                    yield_pct = (div_amount / price_on_date) * 100 if price_on_date > 0 else 0
                else:
                    yield_pct = 0
            except Exception as e:
                logger.warning(f"Could not get price for {symbol} on {div_date}: {e}")
                yield_pct = 0
            
            dividend_yields.append(yield_pct)
        
        dividend_df['dividend_yield_pct'] = dividend_yields
        dividend_df['share_price_on_dividend_date'] = [current_price] * len(dividend_df)
        
        # Add metadata
        company_name = info.get('longName', symbol)
        exchange = info.get('exchange', 'Unknown')
        currency = info.get('currency', 'USD')
        
        # Map exchange to country using the same mapping as the rest of the app
        exchange_country_map = {
            # Index data mappings
            'NASDAQ': 'USA', 'NYSE': 'USA', 'NSE': 'India', 'TSX': 'Canada',
            # High-yield data mappings  
            'ASE': 'USA', 'NGM': 'USA', 'NMS': 'USA', 'NYQ': 'USA',
            'PCX': 'USA', 'TOR': 'Canada',
            # Additional common exchanges
            'LSE': 'UK', 'ASX': 'Australia', 'JPX': 'Japan', 'HKG': 'Hong Kong',
            'SWX': 'Switzerland', 'FRA': 'Germany', 'AMS': 'Netherlands'
        }
        country = exchange_country_map.get(exchange, 'Unknown')
        
        dividend_df['ticker_symbol'] = symbol
        dividend_df['company_name'] = company_name
        dividend_df['exchange'] = exchange
        dividend_df['currency'] = currency
        dividend_df['country'] = country
        
        # Filter to last 10 years - handle timezone-aware datetimes
        ten_years_ago = datetime.now() - timedelta(days=365*10)
        
        # Convert dividend_date to timezone-naive for comparison
        if not dividend_df['dividend_date'].empty:
            # Check if the dividend_date is timezone-aware
            if dividend_df['dividend_date'].dt.tz is not None:
                # Convert to UTC and then remove timezone info for comparison
                dividend_df['dividend_date'] = dividend_df['dividend_date'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        dividend_df = dividend_df[dividend_df['dividend_date'] >= ten_years_ago]
        
        if dividend_df.empty:
            return {
                'success': False,
                'error': f"No dividend data found for '{symbol}' in the last 10 years."
            }
        
        logger.info(f"✅ Successfully fetched {len(dividend_df)} dividend records for {symbol}")
        
        return {
            'success': True,
            'data': dividend_df,
            'symbol': symbol,
            'company_name': company_name,
            'exchange': exchange,
            'currency': currency,
            'record_count': len(dividend_df)
        }
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return {
            'success': False,
            'error': f"Error fetching data for '{symbol}': {str(e)}"
        }

# Custom CSS for loading animation
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .loading-dots {
                display: inline-flex;
                align-items: center;
                color: #FFA500;
            }
            
            .loading-dots::after {
                content: "●●●";
                animation: loading 1.5s infinite;
                margin-left: 5px;
            }
            
            @keyframes loading {
                0% { content: "●○○"; }
                33% { content: "○●○"; }
                66% { content: "○○●"; }
                100% { content: "●○○"; }
            }
            
            .loading-spinner {
                display: inline-block;
                width: 16px;
                height: 16px;
                border: 2px solid #FFA500;
                border-radius: 50%;
                border-top-color: transparent;
                animation: spin 1s linear infinite;
                margin-right: 8px;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# App layout with enhanced 4-dropdown design
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🌍 Global Dividend Analysis Dashboard", 
                   className="text-center mb-4",
                   style={'color': '#fff', 'font-weight': 'bold'})
        ])
    ]),
    
    # Usage Guide (Collapsible)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    dbc.Button(
                        [
                            html.I(className="fas fa-question-circle", style={'margin-right': '8px'}),
                            "📖 How to Use This Dashboard",
                            html.I(id="usage-guide-icon", className="fas fa-chevron-down", style={'margin-left': '10px', 'float': 'right'})
                        ],
                        id="usage-guide-toggle",
                        color="info",
                        outline=True,
                        size="sm",
                        style={'width': '100%', 'text-align': 'left'}
                    )
                ], style={'padding': '10px', 'background-color': '#2c3e50'}),
                dbc.Collapse([
                    dbc.CardBody([
                        html.Div([
                            html.H5("🎯 Quick Start Guide", style={'color': '#3498db', 'margin-bottom': '15px'}),
                            
                            html.Div([
                                html.H6("📊 1. Browse Existing Data", style={'color': '#e74c3c', 'margin-bottom': '8px'}),
                                html.Ul([
                                    html.Li("Use the dropdown filters to explore 200+ pre-loaded stocks"),
                                    html.Li("Filter by Exchange (USA 🇺🇸, India 🇮🇳, Canada 🇨🇦, etc.)"),
                                    html.Li("Filter by Average Yield Range (e.g., 2-4%, 4-6%)"),
                                    html.Li("Select time period for analysis (3, 5, or 10 years)")
                                ], style={'margin-bottom': '15px'})
                            ]),
                            
                            html.Div([
                                html.H6("🔍 2. Add New Stocks", style={'color': '#e74c3c', 'margin-bottom': '8px'}),
                                html.Ul([
                                    html.Li("Type any stock symbol (AAPL, MSFT, TSLA, etc.) in the text box"),
                                    html.Li("Click 'Fetch Data' to get real-time dividend information"),
                                    html.Li("New stocks are automatically added to your dropdown"),
                                    html.Li("Supports global markets: US (.N/A), India (.NS), Canada (.TO)")
                                ], style={'margin-bottom': '15px'})
                            ]),
                            
                            html.Div([
                                html.H6("📈 3. Analyze the Charts", style={'color': '#e74c3c', 'margin-bottom': '8px'}),
                                html.Ul([
                                    html.Li("Main Chart: See dividend trends over time with yield percentages"),
                                    html.Li("Key Metrics Panel: View current yield, total dividends, and consistency score"),
                                    html.Li("Historical Data: Track dividend growth and payment patterns"),
                                    html.Li("Hover over data points for detailed information")
                                ], style={'margin-bottom': '15px'})
                            ]),
                            
                            html.Hr(style={'margin': '20px 0', 'border-color': '#34495e'}),
                            
                            html.Div([
                                html.H6("💡 Pro Tips", style={'color': '#f39c12', 'margin-bottom': '10px'}),
                                html.Ul([
                                    html.Li("🔄 Try different time periods to see long-term vs short-term trends"),
                                    html.Li("🌍 Compare dividend yields across different countries and exchanges"),
                                    html.Li("📊 Use the yield range filter to find high-dividend stocks"),
                                    html.Li("⚡ Charts update automatically when you change selections"),
                                    html.Li("💾 Your added stocks persist during the session")
                                ], style={'margin-bottom': '10px'})
                            ]),
                            
                            html.Div([
                                html.Small([
                                    "💭 ",
                                    html.Strong("Remember: "),
                                    "This dashboard shows historical data for analysis. Always verify information with official sources before making investment decisions."
                                ], style={'color': '#95a5a6', 'font-style': 'italic'})
                            ])
                        ])
                    ])
                ], id="usage-guide-collapse", is_open=False)
            ], style={'margin-bottom': '20px', 'background-color': '#34495e', 'border': 'none'})
        ])
    ]),
    
    # Disclaimer Notice
    dbc.Row([
        dbc.Col([
            dbc.Alert([
                html.H5("⚠️ Important Disclaimer", style={'color': '#856404', 'margin-bottom': '10px'}),
                html.P([
                    "The data presented in this dashboard is for informational purposes only and should ",
                    html.Strong("not be considered ground truth"), 
                    ". Investment decisions should ",
                    html.Strong("never"), 
                    " be made solely based on the values shown here. Always verify all financial data with reliable, official sources before making any investment decisions. Past performance does not guarantee future results."
                ], style={'margin-bottom': '0'})
            ], color="warning", style={'margin-bottom': '20px'})
        ])
    ]),
    
    # Control Panel with 4 Dropdowns
    dbc.Card([
        dbc.CardHeader([
            html.H4("📊 Analysis Controls", style={'color': '#fff', 'margin': '0'})
        ]),
        dbc.CardBody([
            dbc.Row([
                # Dropdown 1: Stock Selection
                dbc.Col([
                    html.Label("🏢 Select Stock:", style={'color': '#fff', 'font-weight': 'bold'}),
                    dcc.Dropdown(
                        id='stock-dropdown',
                        options=stock_options,
                        value=stock_options[0]['value'] if stock_options and not stock_options[0].get('disabled') else None,
                        placeholder="Search for a stock...",
                        style={'backgroundColor': '#333', 'color': '#fff'},
                        className='custom-dropdown'
                    )
                ], width=3),
                
                # Dropdown 2: Exchange Filter
                dbc.Col([
                    html.Label("🏛️ Filter by Exchange:", style={'color': '#fff', 'font-weight': 'bold'}),
                    dcc.Dropdown(
                        id='exchange-dropdown',
                        options=exchange_options,
                        value='ALL',
                        style={'backgroundColor': '#333', 'color': '#fff'},
                        className='custom-dropdown'
                    )
                ], width=2),
                
                # Dropdown 3: Current Annual Yield Filter
                dbc.Col([
                    html.Label("� Current Annual Yield:", style={'color': '#fff', 'font-weight': 'bold'}),
                    dcc.Dropdown(
                        id='yield-dropdown',
                        options=[
                            {'label': 'Any Yield', 'value': 'ANY'},
                            {'label': 'Up to 3%', 'value': '0-3'},
                            {'label': 'Up to 5%', 'value': '0-5'},
                            {'label': 'Up to 10%', 'value': '0-10'},
                            {'label': 'Above 10%', 'value': '10+'}
                        ],
                        value='ANY',
                        style={'backgroundColor': '#333', 'color': '#fff'},
                        className='custom-dropdown'
                    )
                ], width=2),
                
                # Dropdown 4: Timeframe
                dbc.Col([
                    html.Label("📅 Analysis Period:", style={'color': '#fff', 'font-weight': 'bold'}),
                    dcc.Dropdown(
                        id='timeframe-dropdown',
                        options=[
                            {'label': 'Last 10 Years', 'value': 10},
                            {'label': 'Last 5 Years', 'value': 5},
                            {'label': 'Last 3 Years', 'value': 3}
                        ],
                        value=10,
                        style={'backgroundColor': '#333', 'color': '#fff'},
                        className='custom-dropdown'
                    )
                ], width=2),
                
                # Status
                dbc.Col([
                    html.Label("Status:", style={'color': '#fff'}),
                    html.Br(),
                    html.P(f"📊 {len([opt for opt in stock_options if not opt.get('disabled')])} stocks available", 
                           style={'color': '#00CC96', 'font-size': '0.9em'})
                ], width=3)
            ]),
            
            # Second row: Manual Stock Symbol Input
            html.Hr(style={'border-color': '#555', 'margin': '15px 0'}),
            dbc.Row([
                dbc.Col([
                    html.Label("🔍 Add New Stock Symbol:", style={'color': '#fff', 'font-weight': 'bold'}),
                    html.P("Enter any stock symbol to fetch its dividend data dynamically", 
                           style={'color': '#888', 'font-size': '0.85em', 'margin-bottom': '5px'})
                ], width=3),
                
                dbc.Col([
                    dcc.Input(
                        id='manual-stock-input',
                        type='text',
                        value='',
                        placeholder='🔍 Enter stock symbol (e.g., AAPL, MSFT, GOOGL)',
                        style={
                            'width': '100%', 
                            'height': '42px',
                            'backgroundColor': '#2b3440', 
                            'color': '#fff', 
                            'border': '2px solid #4a5568',
                            'borderRadius': '8px',
                            'padding': '0 15px',
                            'fontSize': '14px',
                            'fontWeight': '500',
                            'outline': 'none',
                            'transition': 'all 0.3s ease',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                        },
                        className='custom-input'
                    )
                ], width=3),
                
                dbc.Col([
                    dbc.Button(
                        "🚀 Fetch Data",
                        id='fetch-stock-btn',
                        color='success',
                        style={'width': '100%'}
                    )
                ], width=2),
                
                dbc.Col([
                    dcc.Loading(
                        id="loading-fetch",
                        type="default",
                        children=html.Div(id='fetch-status', style={'color': '#fff', 'padding-top': '8px'}),
                        style={'color': '#FFA500'}
                    )
                ], width=4)
            ])
        ])
    ], style={'backgroundColor': '#333', 'border': '1px solid #555', 'margin-bottom': '20px'}),
    
    # Stock Information Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody(id="stock-info-panel")
            ], style={'backgroundColor': '#333', 'border': '1px solid #555', 'height': '160px'})
        ], width=6, className="pr-1"),
        
        # Key Metrics Panel (Right side of stock info) - Wider now
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("🎯 Key Metrics", style={'color': '#fff', 'margin': '0', 'text-align': 'center'})
                ]),
                dbc.CardBody(id="key-metrics-panel", style={'padding': '10px', 'overflow': 'visible'})
            ], style={'backgroundColor': '#333', 'border': '1px solid #555', 'height': '160px'})
        ], width=6, className="pl-1")
    ], className="mb-3 no-gutters"),
    
    # Charts Section
    dbc.Row([
        # Left Chart: Dividend Yield Timeline
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("� Dividend Yield Timeline", style={'color': '#fff', 'margin': '0'})
                ]),
                dbc.CardBody([
                    dcc.Graph(id="yield-timeline-chart")
                ])
            ], style={'backgroundColor': '#333', 'border': '1px solid #555'})
        ], width=6),
        
        # Right Chart: Dividend Amount & Price
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("💰 Dividend Amount & Stock Price", style={'color': '#fff', 'margin': '0'})
                ]),
                dbc.CardBody([
                    dcc.Graph(id="dividend-amount-chart")
                ])
            ], style={'backgroundColor': '#333', 'border': '1px solid #555'})
        ], width=6)
    ], className="mb-3"),
    
    # Summary Statistics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("📊 Summary Statistics", style={'color': '#fff', 'margin': '0'})
                ]),
                dbc.CardBody(id="summary-stats")
            ], style={'backgroundColor': '#333', 'border': '1px solid #555'})
        ], width=12)
    ]),
    
    # Footer
    html.Hr(style={'border-color': '#555'}),
    dbc.Row([
        dbc.Col([
            html.P("🌍 Global Dividend Analysis Dashboard - Real-time data from major exchanges worldwide", 
                   className="text-center", style={'color': '#888', 'margin-top': '20px', 'margin-bottom': '10px'}),
            html.P([
                "Created by ",
                html.A("Pratik Barve", 
                       href="https://github.com/iamstarstuff",
                       target="_blank",
                       style={'color': '#00CC96', 'text-decoration': 'none', 'font-weight': 'bold'}),
                " | ",
                html.A("GitHub Profile", 
                       href="https://github.com/iamstarstuff",
                       target="_blank",
                       style={'color': '#00CC96', 'text-decoration': 'none'})
            ], className="text-center", style={'color': '#888', 'font-size': '0.9em'})
        ])
    ])
    
], fluid=True, style={'backgroundColor': '#1e1e1e', 'min-height': '100vh', 'padding': '20px'})

# Add custom CSS for dark theme dropdowns
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { 
                background-color: #1e1e1e !important; 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            /* Fix dropdown visibility issues */
            .Select-control {
                background-color: #444 !important;
                border-color: #666 !important;
                color: #fff !important;
            }
            
            .Select-value-label {
                color: #fff !important;
                font-weight: bold !important;
            }
            
            .Select-single-value {
                color: #fff !important;
                font-weight: bold !important;
            }
            
            .Select-input > input {
                color: #fff !important;
            }
            
            .Select-placeholder {
                color: #aaa !important;
            }
            
            .Select-menu-outer {
                background-color: #444 !important;
                border-color: #666 !important;
                z-index: 9999 !important;
            }
            
            .Select-option {
                background-color: #444 !important;
                color: #fff !important;
                padding: 8px 12px !important;
                font-weight: 500 !important;
            }
            
            .Select-option:hover,
            .Select-option.is-focused {
                background-color: #555 !important;
                color: #fff !important;
                font-weight: bold !important;
            }
            
            .Select-option.is-selected {
                background-color: #00CC96 !important;
                color: #fff !important;
                font-weight: bold !important;
            }
            
            .Select-option.is-disabled {
                background-color: #333 !important;
                color: #888 !important;
                font-weight: bold !important;
            }
            
            /* Modern dropdown styling */
            .dash-dropdown {
                font-size: 14px !important;
            }
            
            .dash-dropdown .dropdown {
                background-color: #444 !important;
            }
            
            .dash-dropdown .dropdown .dropdown-toggle {
                background-color: #444 !important;
                border: 2px solid #666 !important;
                color: #fff !important;
                font-weight: bold !important;
                padding: 10px 15px !important;
            }
            
            .dash-dropdown .dropdown .dropdown-toggle:focus {
                border-color: #00CC96 !important;
                box-shadow: 0 0 0 0.2rem rgba(0, 204, 150, 0.25) !important;
            }
            
            .dash-dropdown .dropdown-menu {
                background-color: #444 !important;
                border: 1px solid #666 !important;
                max-height: 300px !important;
                overflow-y: auto !important;
            }
            
            .dash-dropdown .dropdown-item {
                color: #fff !important;
                padding: 8px 15px !important;
                border-bottom: 1px solid #555 !important;
            }
            
            .dash-dropdown .dropdown-item:hover,
            .dash-dropdown .dropdown-item:focus {
                background-color: #555 !important;
                color: #fff !important;
            }
            
            .dash-dropdown .dropdown-item.disabled {
                color: #888 !important;
                background-color: #333 !important;
                font-weight: bold !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

@app.callback(
    Output('stock-dropdown', 'options'),
    [Input('exchange-dropdown', 'value'),
     Input('yield-dropdown', 'value'),
     Input('fetch-status', 'children')]
)
def update_stock_dropdown(selected_exchange, selected_yield, fetch_status):
    """Update stock dropdown based on exchange and yield filters"""
    try:
        # Use cached data with dynamic data instead of loading repeatedly
        df = get_cached_data_with_dynamic()
        
        if df.empty:
            return [{'label': 'No Data Available', 'value': 'none'}]
        
        # Create comprehensive exchange-to-country mapping
        exchange_country_map = {
            # Index data mappings
            'NASDAQ': 'USA', 'NYSE': 'USA', 'NSE': 'India', 'TSX': 'Canada',
            # High-yield data mappings  
            'ASE': 'USA', 'NGM': 'USA', 'NMS': 'USA', 'NYQ': 'USA',  # NYQ is NASDAQ Global Select (US)
            'PCX': 'Pacific Exchange (USA)', 'TOR': 'Canada',
            # Indian Stock Exchanges
            'NSI': 'India',  # National Stock Exchange of India
            'BSE': 'India',  # Bombay Stock Exchange
            # Additional common exchanges
            'LSE': 'UK', 'ASX': 'Australia', 'JPX': 'Japan', 'HKG': 'Hong Kong',
            'SWX': 'Switzerland', 'FRA': 'Germany', 'AMS': 'Netherlands'
        }
        
        # Process data
        df['dividend_date'] = pd.to_datetime(df['dividend_date'])
        df['dividend_per_share'] = pd.to_numeric(df['dividend_per_share'], errors='coerce')
        df['dividend_yield_pct'] = pd.to_numeric(df['dividend_yield_pct'], errors='coerce')
        df = df.dropna(subset=['dividend_per_share', 'dividend_yield_pct'])
        
        # Add country mapping if not present or correct any issues
        if 'country' not in df.columns:
            df['country'] = df['exchange'].map(exchange_country_map).fillna('Unknown')
        else:
            # Update any missing or incorrect country mappings
            df['country'] = df.apply(
                lambda row: exchange_country_map.get(row['exchange'], row.get('country', 'Unknown')), 
                axis=1
            )
        
        # Filter by exchange
        if selected_exchange != 'ALL':
            df = df[df['exchange'] == selected_exchange]
        
        # Calculate statistics per stock using ALL available data (not just recent)
        # This ensures we don't filter out stocks that haven't paid dividends recently
        stock_stats = df.groupby(['ticker_symbol', 'company_name', 'exchange', 'country']).agg({
            'dividend_per_share': ['sum', 'mean'],  # Total and average dividends
            'dividend_yield_pct': 'mean',  # Average historical yield
            'share_price_on_dividend_date': 'last',  # Most recent price
            'dividend_date': 'count'  # Number of dividend records
        }).round(4)
        
        # Flatten column names
        stock_stats.columns = ['total_dividends', 'avg_dividend', 'avg_yield', 'current_price', 'dividend_count']
        stock_stats = stock_stats.reset_index()
        
        # Calculate estimated current annual yield using average dividend frequency
        # For stocks with multiple payments, estimate annual yield
        stock_stats['estimated_annual_yield'] = stock_stats.apply(
            lambda row: (row['avg_dividend'] * min(row['dividend_count'], 4) / row['current_price'] * 100) 
            if row['current_price'] > 0 else row['avg_yield'], 
            axis=1
        ).round(2)
        
        # Filter by yield range using estimated annual yield
        if selected_yield != 'ANY':
            if selected_yield == '0-3':
                stock_stats = stock_stats[stock_stats['estimated_annual_yield'] <= 3]
            elif selected_yield == '0-5':
                stock_stats = stock_stats[stock_stats['estimated_annual_yield'] <= 5]
            elif selected_yield == '0-10':
                stock_stats = stock_stats[stock_stats['estimated_annual_yield'] <= 10]
            elif selected_yield == '10+':
                stock_stats = stock_stats[stock_stats['estimated_annual_yield'] > 10]
        
        # Filter for stocks with at least 3 dividend records
        stock_stats = stock_stats[stock_stats['dividend_count'] >= 3]
        
        # Sort by exchange, then by estimated annual yield descending
        stock_stats = stock_stats.sort_values(['exchange', 'estimated_annual_yield'], ascending=[True, False])
        
        # Create dropdown options with country info
        options = []
        current_exchange = None
        
        # Country flag mapping
        country_flags = {
            'USA': '🇺🇸', 'India': '🇮🇳', 'Canada': '🇨🇦', 'UK': '🇬🇧', 
            'Australia': '🇦🇺', 'Japan': '🇯🇵', 'Hong Kong': '🇭🇰',
            'South Korea': '🇰🇷', 'Switzerland': '🇨🇭', 'Germany': '🇩🇪', 
            'Netherlands': '🇳🇱', 'Pacific Exchange (USA)': '🇺🇸'
        }
        
        for _, row in stock_stats.iterrows():
            # Add exchange headers for multi-exchange view with country names
            if selected_exchange == 'ALL' and row['exchange'] != current_exchange:
                if current_exchange is not None:
                    options.append({'label': '─' * 60, 'value': 'separator', 'disabled': True})
                
                flag = country_flags.get(row['country'], '🌍')
                options.append({
                    'label': f"{flag} {row['exchange']} - {row['country']}",
                    'value': f"header_{row['exchange']}",
                    'disabled': True
                })
                current_exchange = row['exchange']
            
            # Add stock option with estimated annual yield
            yield_indicator = "🔥" if row['estimated_annual_yield'] > 10 else ""
            label = f"{yield_indicator}{row['ticker_symbol']} - {row['company_name'][:40]}{'...' if len(row['company_name']) > 40 else ''} (Est: {row['estimated_annual_yield']:.1f}%)"
            options.append({
                'label': label,
                'value': row['ticker_symbol']
            })
        
        if not options:
            return [{'label': 'No stocks match your criteria', 'value': 'none'}]
        
        return options
    except Exception as e:
        logger.error(f"Error updating stock dropdown: {e}")
        return [{'label': 'Error Loading Data', 'value': 'none'}]

# Callback for stock information panel
@app.callback(
    Output('stock-info-panel', 'children'),
    [Input('stock-dropdown', 'value')]
)
def update_stock_info(selected_stock):
    """Update stock information panel"""
    if not selected_stock or selected_stock == 'none' or selected_stock.startswith('header_'):
        return html.P("Select a stock to view information", style={'color': '#888'})
    
    try:
        # Use the cached data with dynamic data
        df = get_cached_data_with_dynamic()
        
        # Filter for selected stock
        stock_data = df[df['ticker_symbol'] == selected_stock]
        
        if stock_data.empty:
            return html.P("No data available for selected stock", style={'color': '#ff6b6b'})
        
        # Process data - make a copy to avoid SettingWithCopyWarning
        stock_data = stock_data.copy()
        stock_data.loc[:, 'dividend_date'] = pd.to_datetime(stock_data['dividend_date'])
        stock_data.loc[:, 'dividend_per_share'] = pd.to_numeric(stock_data['dividend_per_share'], errors='coerce')
        stock_data.loc[:, 'dividend_yield_pct'] = pd.to_numeric(stock_data['dividend_yield_pct'], errors='coerce')
        stock_data = stock_data.dropna(subset=['dividend_per_share', 'dividend_yield_pct'])
        
        if stock_data.empty:
            return html.P("No valid dividend data for selected stock", style={'color': '#ff6b6b'})
        
        # Calculate statistics
        company_name = stock_data['company_name'].iloc[0]
        exchange = stock_data['exchange'].iloc[0]
        country = stock_data['country'].iloc[0] if 'country' in stock_data.columns else 'Unknown'
        currency = stock_data['currency'].iloc[0] if 'currency' in stock_data.columns else 'USD'
        avg_yield = stock_data['dividend_yield_pct'].mean()
        total_dividends = len(stock_data)
        latest_dividend = stock_data['dividend_per_share'].iloc[-1] if not stock_data.empty else 0
        latest_date = stock_data['dividend_date'].max()
        
        # Calculate annualized yield using the SAME method as dropdown for consistency
        # Use average dividend and frequency estimation (same as dropdown logic)
        avg_dividend = stock_data['dividend_per_share'].mean()
        dividend_count = len(stock_data)
        
        # Use the most recent stock price from our data
        latest_price_data = stock_data.iloc[-1] if not stock_data.empty else None
        if latest_price_data is not None and 'share_price_on_dividend_date' in stock_data.columns:
            recent_price = latest_price_data['share_price_on_dividend_date']
            if recent_price > 0 and avg_dividend > 0:
                # Use SAME calculation as dropdown: avg_dividend * min(dividend_count, 4) / current_price * 100
                annual_yield = (avg_dividend * min(dividend_count, 4) / recent_price) * 100
            else:
                annual_yield = None
        else:
            annual_yield = None
        
        # Currency symbols mapping
        currency_symbols = {
            'USD': '$', 'CAD': 'C$', 'GBP': '£', 'EUR': '€', 
            'AUD': 'A$', 'INR': '₹', 'JPY': '¥', 'CNY': '¥',
            'HKD': 'HK$', 'SGD': 'S$', 'CHF': 'CHF', 'SEK': 'kr',
            'NOK': 'kr', 'DKK': 'kr', 'PLN': 'zł', 'CZK': 'Kč'
        }
        currency_symbol = currency_symbols.get(currency, currency + ' ')
        
        return dbc.Row([
            dbc.Col([
                html.H5(f"🏢 {company_name}", style={'color': '#00CC96', 'margin-bottom': '10px'}),
                html.P(f"📍 {exchange} ({country})", style={'color': '#fff', 'margin-bottom': '5px'}),
                html.P(f"🎯 Symbol: {selected_stock}", style={'color': '#fff', 'margin-bottom': '5px'})
            ], width=6),
            dbc.Col([
                html.P(f"📈 Avg Historical Yield: {avg_yield:.2f}%", style={'color': '#fff', 'margin-bottom': '5px'}),
                html.P(f"💰 Latest Dividend: {currency_symbol}{latest_dividend:.3f}", style={'color': '#fff', 'margin-bottom': '5px'}),
                html.P(f"📅 Last Payment: {latest_date.strftime('%Y-%m-%d')}", style={'color': '#fff', 'margin-bottom': '5px'}),
                html.P(f"📊 Total Records: {total_dividends}", style={'color': '#fff', 'margin-bottom': '5px'})
            ], width=6)
        ])
    except Exception as e:
        logger.error(f"Error updating stock info: {e}")
        return html.P(f"Error loading stock information: {str(e)}", style={'color': '#ff6b6b'})

# Callback for key metrics panel
@app.callback(
    Output('key-metrics-panel', 'children'),
    [Input('stock-dropdown', 'value'),
     Input('timeframe-dropdown', 'value')]
)
def update_key_metrics(selected_stock, timeframe_years):
    """Update key metrics panel with the three main values"""
    if not selected_stock or selected_stock == 'none' or selected_stock.startswith('header_'):
        return html.Div([
            html.P("Select a stock to view key metrics", style={'color': '#888', 'text-align': 'center'})
        ])
    
    try:
        # Use cached data including dynamic stocks
        df = get_cached_data_with_dynamic()
        
        # Filter for selected stock
        stock_data = df[df['ticker_symbol'] == selected_stock].copy()
        stock_data['dividend_date'] = pd.to_datetime(stock_data['dividend_date'])
        
        # Filter by timeframe for TTM calculation
        cutoff_date = datetime.now() - timedelta(days=365 * timeframe_years)
        stock_data = stock_data[stock_data['dividend_date'] >= cutoff_date]
        
        stock_data['dividend_per_share'] = pd.to_numeric(stock_data['dividend_per_share'], errors='coerce')
        stock_data['dividend_yield_pct'] = pd.to_numeric(stock_data['dividend_yield_pct'], errors='coerce')
        stock_data = stock_data.dropna(subset=['dividend_per_share', 'dividend_yield_pct'])
        
        if stock_data.empty:
            return html.P("No data available", style={'color': '#ff6b6b', 'text-align': 'center'})
        
        # Get currency information
        currency = stock_data['currency'].iloc[0] if 'currency' in stock_data.columns else 'USD'
        currency_symbols = {
            'USD': '$', 'CAD': 'C$', 'GBP': '£', 'EUR': '€', 
            'AUD': 'A$', 'INR': '₹', 'JPY': '¥', 'CNY': '¥',
            'HKD': 'HK$', 'SGD': 'S$', 'CHF': 'CHF', 'SEK': 'kr',
            'NOK': 'kr', 'DKK': 'kr', 'PLN': 'zł', 'CZK': 'Kč'
        }
        currency_symbol = currency_symbols.get(currency, currency + ' ')
        
        # Calculate the three key metrics
        
        # 1. Estimated Annual Yield (using same method as stock info panel)
        avg_dividend = stock_data['dividend_per_share'].mean()
        dividend_count = len(stock_data)
        latest_price_data = stock_data.iloc[-1] if not stock_data.empty else None
        estimated_annual_yield = None
        
        if latest_price_data is not None and 'share_price_on_dividend_date' in stock_data.columns:
            recent_price = latest_price_data['share_price_on_dividend_date']
            if recent_price > 0 and avg_dividend > 0:
                estimated_annual_yield = (avg_dividend * min(dividend_count, 4) / recent_price) * 100
        
        # 2. Current Annual Yield (TTM method)
        recent_dividends = stock_data[stock_data['dividend_date'] >= (datetime.now() - timedelta(days=365))]
        annual_dividend_total = recent_dividends['dividend_per_share'].sum() if not recent_dividends.empty else 0
        current_annual_yield = None
        
        if latest_price_data is not None and 'share_price_on_dividend_date' in stock_data.columns:
            recent_price = latest_price_data['share_price_on_dividend_date']
            if recent_price > 0 and annual_dividend_total > 0:
                current_annual_yield = (annual_dividend_total / recent_price) * 100
        
        # 3. Last 12 Months Dividends
        annual_dividends_amount = annual_dividend_total
        
        return html.Div([
            # Three metrics side by side
            html.Div([
                # Estimated Annual Yield Square
                html.Div([
                    html.Div(f"{estimated_annual_yield:.1f}%" if estimated_annual_yield else "N/A", 
                           style={'color': '#fff', 'font-weight': 'bold', 'font-size': '1.4em', 
                                  'display': 'flex', 'align-items': 'center', 'justify-content': 'center',
                                  'height': '60px', 'width': '100%', 'background-color': '#1f4e79',
                                  'border': '2px solid #00CC96', 'border-radius': '8px'}),
                    html.P("🎯 Estimated Annual Yield", 
                          style={'color': '#00CC96', 'font-size': '0.7em', 'text-align': 'center', 
                                 'margin-top': '5px', 'margin-bottom': '0', 'line-height': '1.1'})
                ], style={'flex': '1', 'margin-right': '5px'}),
                
                # Current Yield Square  
                html.Div([
                    html.Div(f"{current_annual_yield:.2f}%" if current_annual_yield else "N/A", 
                           style={'color': '#fff', 'font-weight': 'bold', 'font-size': '1.4em',
                                  'display': 'flex', 'align-items': 'center', 'justify-content': 'center',
                                  'height': '60px', 'width': '100%', 'background-color': '#4a4a00',
                                  'border': '2px solid #FFA500', 'border-radius': '8px'}),
                    html.P("📈 Current Yield", 
                          style={'color': '#FFA500', 'font-size': '0.7em', 'text-align': 'center',
                                 'margin-top': '5px', 'margin-bottom': '0', 'line-height': '1.1'})
                ], style={'flex': '1', 'margin-left': '5px', 'margin-right': '5px'}),
                
                # Last 12 Months Dividend Square
                html.Div([
                    html.Div(f"{currency_symbol}{annual_dividends_amount:.2f}" if annual_dividends_amount > 0 else "N/A", 
                           style={'color': '#fff', 'font-weight': 'bold', 'font-size': '1.2em',
                                  'display': 'flex', 'align-items': 'center', 'justify-content': 'center',
                                  'height': '60px', 'width': '100%', 'background-color': '#0d4f5c',
                                  'border': '2px solid #17A2B8', 'border-radius': '8px'}),
                    html.P("💰 Last 12 Months Dividend", 
                          style={'color': '#17A2B8', 'font-size': '0.7em', 'text-align': 'center',
                                 'margin-top': '5px', 'margin-bottom': '0', 'line-height': '1.1'})
                ], style={'flex': '1', 'margin-left': '5px'})
                
            ], style={'display': 'flex', 'width': '100%', 'height': '100%', 'align-items': 'flex-start'})
        ], style={'width': '100%', 'padding': '0', 'height': '100%'})
        
    except Exception as e:
        logger.error(f"Error updating key metrics: {e}")
        return html.P(f"Error loading metrics: {str(e)}", style={'color': '#ff6b6b', 'text-align': 'center'})

# Callback for yield timeline chart
@app.callback(
    Output('yield-timeline-chart', 'figure'),
    [Input('stock-dropdown', 'value'),
     Input('timeframe-dropdown', 'value')]
)
def update_yield_timeline(selected_stock, timeframe_years):
    """Update dividend yield timeline chart"""
    if not selected_stock or selected_stock == 'none' or selected_stock.startswith('header_'):
        return {
            'data': [],
            'layout': {
                'title': 'Select a stock to view yield timeline',
                'paper_bgcolor': '#333',
                'plot_bgcolor': '#333',
                'font': {'color': '#fff'}
            }
        }
    
    try:
        # Use cached data instead of loading repeatedly
        df = get_cached_data_with_dynamic()
        
        # Filter for selected stock and timeframe
        stock_data = df[df['ticker_symbol'] == selected_stock].copy()
        stock_data['dividend_date'] = pd.to_datetime(stock_data['dividend_date'])
        
        # Filter by timeframe
        cutoff_date = datetime.now() - timedelta(days=365 * timeframe_years)
        stock_data = stock_data[stock_data['dividend_date'] >= cutoff_date]
        
        stock_data['dividend_yield_pct'] = pd.to_numeric(stock_data['dividend_yield_pct'], errors='coerce')
        stock_data = stock_data.dropna(subset=['dividend_yield_pct'])
        stock_data = stock_data.sort_values('dividend_date')
        
        if stock_data.empty:
            return {
                'data': [],
                'layout': {
                    'title': f'No data available for {selected_stock} in last {timeframe_years} years',
                    'paper_bgcolor': '#333',
                    'plot_bgcolor': '#333',
                    'font': {'color': '#fff'}
                }
            }
        
        # Create timeline chart
        fig = go.Figure()
        
        # Add yield line
        fig.add_trace(go.Scatter(
            x=stock_data['dividend_date'],
            y=stock_data['dividend_yield_pct'],
            mode='lines+markers',
            name='Dividend Yield',
            line=dict(color='#00CC96', width=3),
            marker=dict(size=6, color='#00CC96'),
            hovertemplate='<b>%{x}</b><br>Yield: %{y:.2f}%<extra></extra>'
        ))
        
        # Add average line
        avg_yield = stock_data['dividend_yield_pct'].mean()
        fig.add_hline(
            y=avg_yield,
            line_dash="dash",
            line_color="#FFA500",
            annotation_text=f"Average: {avg_yield:.2f}%",
            annotation_position="top right"
        )
        
        fig.update_layout(
            title=f'📈 Dividend Yield Timeline - {selected_stock} (Last {timeframe_years} Years)',
            xaxis_title='Date',
            yaxis_title='Dividend Yield (%)',
            paper_bgcolor='#333',
            plot_bgcolor='#333',
            font=dict(color='#fff'),
            hovermode='x unified',
            showlegend=False
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating yield timeline: {e}")
        return {
            'data': [],
            'layout': {
                'title': f'Error loading data for {selected_stock}',
                'paper_bgcolor': '#333',
                'plot_bgcolor': '#333',
                'font': {'color': '#fff'}
            }
        }

# Callback for dividend amount chart
@app.callback(
    Output('dividend-amount-chart', 'figure'),
    [Input('stock-dropdown', 'value'),
     Input('timeframe-dropdown', 'value')]
)
def update_dividend_amount_chart(selected_stock, timeframe_years):
    """Update dividend amount and stock price chart"""
    if not selected_stock or selected_stock == 'none' or selected_stock.startswith('header_'):
        return {
            'data': [],
            'layout': {
                'title': 'Select a stock to view dividend amounts',
                'paper_bgcolor': '#333',
                'plot_bgcolor': '#333',
                'font': {'color': '#fff'}
            }
        }
    
    try:
        # Use cached data instead of loading repeatedly
        df = get_cached_data_with_dynamic()
        
        # Filter for selected stock and timeframe
        stock_data = df[df['ticker_symbol'] == selected_stock].copy()
        
        if stock_data.empty:
            return {
                'data': [],
                'layout': {
                    'title': f'No data found for {selected_stock}',
                    'paper_bgcolor': '#333',
                    'plot_bgcolor': '#333',
                    'font': {'color': '#fff'}
                }
            }
        
        # Process dates and numeric data
        stock_data['dividend_date'] = pd.to_datetime(stock_data['dividend_date'])
        
        # Get currency information
        currency = stock_data['currency'].iloc[0] if 'currency' in stock_data.columns and not stock_data.empty else 'USD'
        currency_symbols = {
            'USD': '$', 'CAD': 'C$', 'GBP': '£', 'EUR': '€', 
            'AUD': 'A$', 'INR': '₹', 'JPY': '¥', 'CNY': '¥',
            'HKD': 'HK$', 'SGD': 'S$', 'CHF': 'CHF', 'SEK': 'kr',
            'NOK': 'kr', 'DKK': 'kr', 'PLN': 'zł', 'CZK': 'Kč'
        }
        currency_symbol = currency_symbols.get(currency, currency + ' ')
        
        # Filter by timeframe
        cutoff_date = datetime.now() - timedelta(days=365 * timeframe_years)
        stock_data = stock_data[stock_data['dividend_date'] >= cutoff_date]
        
        # Convert dividend amounts to numeric
        stock_data['dividend_per_share'] = pd.to_numeric(stock_data['dividend_per_share'], errors='coerce')
        
        # Remove rows with invalid dividend data
        valid_dividend_data = stock_data.dropna(subset=['dividend_per_share'])
        valid_dividend_data = valid_dividend_data[valid_dividend_data['dividend_per_share'] > 0]
        valid_dividend_data = valid_dividend_data.sort_values('dividend_date')
        
        if valid_dividend_data.empty:
            return {
                'data': [],
                'layout': {
                    'title': f'No valid dividend data for {selected_stock} in last {timeframe_years} years',
                    'paper_bgcolor': '#333',
                    'plot_bgcolor': '#333',
                    'font': {'color': '#fff'},
                    'annotations': [{
                        'text': f'Available data: {len(stock_data)} records<br>Valid dividends: {len(valid_dividend_data)} records',
                        'x': 0.5,
                        'y': 0.5,
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'color': '#fff', 'size': 14}
                    }]
                }
            }
        
        # Create dual-axis chart
        fig = make_subplots(
            specs=[[{"secondary_y": True}]],
            subplot_titles=[f'📊 Found {len(valid_dividend_data)} dividend payments']
        )
        
        # Add dividend amount bars
        fig.add_trace(
            go.Bar(
                x=valid_dividend_data['dividend_date'],
                y=valid_dividend_data['dividend_per_share'],
                name='Dividend per Share',
                marker_color='#00CC96',
                hovertemplate=f'<b>%{{x}}</b><br>Dividend: {currency_symbol}%{{y:.4f}}<extra></extra>',
                opacity=0.8
            ),
            secondary_y=False
        )
        
        # Try to add stock price line if available
        # Check for different possible column names for stock price
        price_column = None
        if 'stock_price' in stock_data.columns:
            price_column = 'stock_price'
        elif 'share_price_on_dividend_date' in stock_data.columns:
            price_column = 'share_price_on_dividend_date'
        elif 'price' in stock_data.columns:
            price_column = 'price'
        
        if price_column:
            stock_data.loc[:, 'stock_price'] = pd.to_numeric(stock_data[price_column], errors='coerce')
            valid_price_data = stock_data.dropna(subset=['stock_price'])
            valid_price_data = valid_price_data[valid_price_data['stock_price'] > 0]
            
            if not valid_price_data.empty:
                valid_price_data = valid_price_data.sort_values('dividend_date')
                fig.add_trace(
                    go.Scatter(
                        x=valid_price_data['dividend_date'],
                        y=valid_price_data['stock_price'],
                        mode='lines+markers',
                        name='Stock Price',
                        line=dict(color='#FFA500', width=2),
                        marker=dict(size=4),
                        hovertemplate=f'<b>%{{x}}</b><br>Price: {currency_symbol}%{{y:.2f}}<extra></extra>'
                    ),
                    secondary_y=True
                )
                
                # Update y-axes titles
                fig.update_yaxes(title_text=f"Dividend per Share ({currency})", secondary_y=False, color='#00CC96')
                fig.update_yaxes(title_text=f"Stock Price ({currency})", secondary_y=True, color='#FFA500')
            else:
                # Only dividend data available
                fig.update_yaxes(title_text=f"Dividend per Share ({currency})", secondary_y=False, color='#00CC96')
        else:
            # Only dividend data available
            fig.update_yaxes(title_text=f"Dividend per Share ({currency})", secondary_y=False, color='#00CC96')
        
        # Update layout
        fig.update_layout(
            title=f'💰 Dividend History - {selected_stock} (Last {timeframe_years} Years)',
            paper_bgcolor='#333',
            plot_bgcolor='#333',
            font=dict(color='#fff'),
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                bgcolor='rgba(68,68,68,0.8)',
                bordercolor='#666',
                borderwidth=1
            ),
            xaxis=dict(
                title='Date',
                color='#fff',
                gridcolor='#555'
            ),
            margin=dict(t=80, b=60, l=60, r=60)
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating dividend amount chart: {e}")
        return {
            'data': [],
            'layout': {
                'title': f'Error loading data for {selected_stock}',
                'paper_bgcolor': '#333',
                'plot_bgcolor': '#333',
                'font': {'color': '#fff'},
                'annotations': [{
                    'text': f'Error: {str(e)}',
                    'x': 0.5,
                    'y': 0.5,
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'color': '#ff6b6b', 'size': 14}
                }]
            }
        }

# Callback for summary statistics
@app.callback(
    Output('summary-stats', 'children'),
    [Input('stock-dropdown', 'value'),
     Input('timeframe-dropdown', 'value')]
)
def update_summary_stats(selected_stock, timeframe_years):
    """Update summary statistics panel"""
    if not selected_stock or selected_stock == 'none' or selected_stock.startswith('header_'):
        return html.P("Select a stock to view summary statistics", style={'color': '#888'})
    
    try:
        # Use cached data instead of loading repeatedly
        df = get_cached_data_with_dynamic()
        
        # Filter for selected stock and timeframe
        stock_data = df[df['ticker_symbol'] == selected_stock].copy()
        stock_data['dividend_date'] = pd.to_datetime(stock_data['dividend_date'])
        
        # Filter by timeframe
        cutoff_date = datetime.now() - timedelta(days=365 * timeframe_years)
        stock_data = stock_data[stock_data['dividend_date'] >= cutoff_date]
        
        stock_data['dividend_per_share'] = pd.to_numeric(stock_data['dividend_per_share'], errors='coerce')
        stock_data['dividend_yield_pct'] = pd.to_numeric(stock_data['dividend_yield_pct'], errors='coerce')
        stock_data = stock_data.dropna(subset=['dividend_per_share', 'dividend_yield_pct'])
        
        if stock_data.empty:
            return html.P("No data available for analysis", style={'color': '#ff6b6b'})
        
        # Get currency information
        currency = stock_data['currency'].iloc[0] if 'currency' in stock_data.columns else 'USD'
        currency_symbols = {
            'USD': '$', 'CAD': 'C$', 'GBP': '£', 'EUR': '€', 
            'AUD': 'A$', 'INR': '₹', 'JPY': '¥', 'CNY': '¥',
            'HKD': 'HK$', 'SGD': 'S$', 'CHF': 'CHF', 'SEK': 'kr',
            'NOK': 'kr', 'DKK': 'kr', 'PLN': 'zł', 'CZK': 'Kč'
        }
        currency_symbol = currency_symbols.get(currency, currency + ' ')
        
        # Calculate statistics
        total_dividends = stock_data['dividend_per_share'].sum()
        avg_dividend = stock_data['dividend_per_share'].mean()
        max_dividend = stock_data['dividend_per_share'].max()
        min_dividend = stock_data['dividend_per_share'].min()
        avg_yield = stock_data['dividend_yield_pct'].mean()
        max_yield = stock_data['dividend_yield_pct'].max()
        min_yield = stock_data['dividend_yield_pct'].min()
        num_payments = len(stock_data)
        
        # Calculate annual yield using recent 12 months of dividends (fast method)
        recent_dividends = stock_data[stock_data['dividend_date'] >= (datetime.now() - timedelta(days=365))]
        annual_dividend_total = recent_dividends['dividend_per_share'].sum() if not recent_dividends.empty else 0
        
        # Use most recent stock price from our data instead of live API call
        latest_price_data = stock_data.iloc[-1] if not stock_data.empty else None
        annual_yield = None
        if latest_price_data is not None and 'share_price_on_dividend_date' in stock_data.columns:
            recent_price = latest_price_data['share_price_on_dividend_date']
            if recent_price > 0 and annual_dividend_total > 0:
                annual_yield = (annual_dividend_total / recent_price) * 100
        
        # Calculate annual totals
        stock_data.loc[:, 'year'] = stock_data['dividend_date'].dt.year
        annual_dividends = stock_data.groupby('year')['dividend_per_share'].sum()
        best_year = annual_dividends.idxmax() if not annual_dividends.empty else 'N/A'
        best_year_amount = annual_dividends.max() if not annual_dividends.empty else 0
        
        return dbc.Row([
            dbc.Col([
                html.H6("💰 Dividend Amounts", style={'color': '#00CC96', 'margin-bottom': '15px'}),
                html.P(f"Total Dividends: {currency_symbol}{total_dividends:.3f}", style={'color': '#fff', 'margin-bottom': '5px'}),
                html.P(f"Average per Payment: {currency_symbol}{avg_dividend:.3f}", style={'color': '#fff', 'margin-bottom': '5px'}),
                html.P(f"Highest Payment: {currency_symbol}{max_dividend:.3f}", style={'color': '#fff', 'margin-bottom': '5px'}),
                html.P(f"Lowest Payment: {currency_symbol}{min_dividend:.3f}", style={'color': '#fff', 'margin-bottom': '5px'}),
            ], width=3),
            dbc.Col([
                html.H6("📈 Yield Statistics", style={'color': '#FFA500', 'margin-bottom': '15px'}),
                html.P(f"Average Historical Yield: {avg_yield:.2f}%", style={'color': '#fff', 'margin-bottom': '5px'}),
                html.P(f"Highest Yield: {max_yield:.2f}%", style={'color': '#fff', 'margin-bottom': '5px'}),
                html.P(f"Lowest Yield: {min_yield:.2f}%", style={'color': '#fff', 'margin-bottom': '5px'}),
                html.P(f"Yield Range: {max_yield - min_yield:.2f}%", style={'color': '#fff', 'margin-bottom': '5px'}),
            ], width=3),
            dbc.Col([
                html.H6("📊 Payment History", style={'color': '#17A2B8', 'margin-bottom': '15px'}),
                html.P(f"Total Payments: {num_payments}", style={'color': '#fff', 'margin-bottom': '5px'}),
                html.P(f"Analysis Period: {timeframe_years} years", style={'color': '#fff', 'margin-bottom': '5px'}),
                html.P(f"Avg per Year: {num_payments / timeframe_years:.1f}", style={'color': '#fff', 'margin-bottom': '5px'}),
                html.P(f"Best Year: {best_year} ({currency_symbol}{best_year_amount:.3f})", style={'color': '#fff', 'margin-bottom': '5px'}),
            ], width=3),
            dbc.Col([
                html.H6("🎯 Investment Insights", style={'color': '#E83E8C', 'margin-bottom': '15px'}),
                html.P(f"Consistency Score: {100 - (stock_data['dividend_yield_pct'].std() / avg_yield * 100):.0f}%", 
                       style={'color': '#fff', 'margin-bottom': '5px'}),
                html.P(f"Growth Trend: {'📈 Positive' if stock_data['dividend_per_share'].iloc[-1] > stock_data['dividend_per_share'].iloc[0] else '📉 Negative'}", 
                       style={'color': '#fff', 'margin-bottom': '5px'}),
                html.P(f"Risk Level: {'Low' if stock_data['dividend_yield_pct'].std() < 2 else 'Medium' if stock_data['dividend_yield_pct'].std() < 4 else 'High'}", 
                       style={'color': '#fff', 'margin-bottom': '5px'}),
                html.P(f"Currency: {currency}", style={'color': '#888', 'margin-bottom': '5px', 'font-size': '0.85em'}),
            ], width=3)
        ])
    except Exception as e:
        logger.error(f"Error calculating summary stats: {e}")
        return html.P(f"Error calculating statistics: {str(e)}", style={'color': '#ff6b6b'})

# Callback for manual stock symbol fetching
@app.callback(
    [Output('fetch-status', 'children'),
     Output('manual-stock-input', 'value')],
    [Input('fetch-stock-btn', 'n_clicks')],
    [dash.dependencies.State('manual-stock-input', 'value')]
)
def fetch_new_stock_data(n_clicks, stock_symbol):
    """Handle manual stock symbol input and fetch its dividend data"""
    global DYNAMIC_STOCK_DATA, LAST_FETCHED_SYMBOL
    
    if n_clicks is None or not stock_symbol:
        # Return current state without changes
        return ("", "")
    
    try:
        # Clean the symbol (in case it's typed manually)
        symbol = str(stock_symbol).strip().upper()
        
        # Fetch the stock data
        result = fetch_stock_dividend_data(symbol)
        
        if result['success']:
            # Store the fetched data globally
            symbol = result['symbol']
            DYNAMIC_STOCK_DATA[symbol] = result['data']
            LAST_FETCHED_SYMBOL = symbol  # Store for auto-selection
            
            # Success status with enhanced animation
            status_success = html.Div([
                html.I(className="fas fa-check-circle", style={'margin-right': '8px', 'color': '#00CC96'}),
                html.Span(f"✅ Successfully fetched {result['record_count']} dividend records for {symbol}!", 
                         style={'color': '#00CC96', 'font-weight': 'bold'})
            ], style={'display': 'flex', 'align-items': 'center'})
            
            return status_success, None  # Clear input
            
        else:
            # Error status
            status_error = html.Div([
                html.I(className="fas fa-exclamation-triangle", style={'margin-right': '8px', 'color': '#ff6b6b'}),
                html.Span(result['error'], style={'color': '#ff6b6b'})
            ], style={'display': 'flex', 'align-items': 'center'})
            
            return status_error, stock_symbol  # Keep input for retry
            
    except Exception as e:
        logger.error(f"Error in fetch callback: {e}")
        status_error = html.Div([
            html.I(className="fas fa-exclamation-triangle", style={'margin-right': '8px', 'color': '#ff6b6b'}),
            html.Span(f"❌ Error: {str(e)}", style={'color': '#ff6b6b'})
        ], style={'display': 'flex', 'align-items': 'center'})
        
        return status_error, stock_symbol

# Auto-selection callback - triggered when stock dropdown options are updated
@app.callback(
    Output('stock-dropdown', 'value'),
    [Input('stock-dropdown', 'options')]
)
def auto_select_fetched_stock(dropdown_options):
    """Auto-select the newly fetched stock in the dropdown"""
    global LAST_FETCHED_SYMBOL
    
    if not LAST_FETCHED_SYMBOL or not dropdown_options:
        return dash.no_update
    
    # Check if the last fetched symbol is now available in dropdown options
    available_symbols = [opt['value'] for opt in dropdown_options if not opt.get('disabled')]
    
    if LAST_FETCHED_SYMBOL in available_symbols:
        symbol_to_select = LAST_FETCHED_SYMBOL
        LAST_FETCHED_SYMBOL = None  # Clear after using
        logger.info(f"🎯 Auto-selecting fetched stock: {symbol_to_select}")
        return symbol_to_select
    
    return dash.no_update

# Callback for usage guide toggle
@app.callback(
    [Output('usage-guide-collapse', 'is_open'),
     Output('usage-guide-icon', 'className')],
    [Input('usage-guide-toggle', 'n_clicks')],
    [State('usage-guide-collapse', 'is_open')]
)
def toggle_usage_guide(n_clicks, is_open):
    """Toggle the usage guide visibility and update the icon"""
    if n_clicks:
        new_state = not is_open
        icon_class = "fas fa-chevron-up" if new_state else "fas fa-chevron-down"
        return new_state, icon_class
    
    return is_open, "fas fa-chevron-down"

# Modified get_cached_data function to include dynamic data
if __name__ == '__main__':
    # For deployment on Render or other cloud platforms
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host='0.0.0.0', port=port)
