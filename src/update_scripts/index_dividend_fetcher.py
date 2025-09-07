"""
Dynamic Index-Based Dividend Data Fetcher
Fetches top companies from major global indices for dividend analysis
"""

import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
import logging
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndexBasedDividendFetcher:
    """Fetches dividend data from major global indices"""
    
    def __init__(self):
        self.indices_config = {
            'SP500': {
                'name': 'S&P 500',
                'country': 'USA',
                'currency': 'USD',
                'exchange': 'NYSE/NASDAQ',
                'top_count': 50,
                'url': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                'table_id': 0
            },
            'NIFTY50': {
                'name': 'NIFTY 50',
                'country': 'India',
                'currency': 'INR',
                'exchange': 'NSE',
                'top_count': 50,
                'symbols': [
                    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
                    'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS',
                    'LT.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'MARUTI.NS', 'TITAN.NS',
                    'NESTLEIND.NS', 'HCLTECH.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'WIPRO.NS',
                    'BAJFINANCE.NS', 'ONGC.NS', 'TECHM.NS', 'TATAMOTORS.NS', 'POWERGRID.NS',
                    'NTPC.NS', 'JSWSTEEL.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'INDUSINDBK.NS',
                    'BAJAJFINSV.NS', 'GRASIM.NS', 'CIPLA.NS', 'COALINDIA.NS', 'EICHERMOT.NS',
                    'BRITANNIA.NS', 'HEROMOTOCO.NS', 'BPCL.NS', 'TATASTEEL.NS', 'APOLLOHOSP.NS',
                    'HDFCLIFE.NS', 'SBILIFE.NS', 'ADANIPORTS.NS', 'UPL.NS', 'BAJAJ-AUTO.NS',
                    'TATACONSUM.NS', 'HINDALCO.NS', 'ADANIENT.NS', 'SHREECEM.NS', 'GODREJCP.NS'
                ]
            },
            'TSX60': {
                'name': 'TSX 60',
                'country': 'Canada',
                'currency': 'CAD',
                'exchange': 'TSX',
                'top_count': 30,
                'symbols': [
                    'RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO',
                    'ENB.TO', 'TRP.TO', 'SU.TO', 'CNQ.TO', 'FNV.TO', 
                    'ABX.TO', 'NTR.TO', 'POW.TO', 'MFC.TO', 'SLF.TO', 
                    'FSV.TO', 'CVE.TO', 'AEM.TO', 'K.TO', 'CP.TO',
                    'QSR.TO', 'WSP.TO', 'ATD.TO', 'CNR.TO', 'IMO.TO',
                    'BEP-UN.TO', 'CCO.TO', 'WPM.TO', 'CSU.TO', 'WCN.TO'
                ]
            }
        }
        
        # Mapping for exchange standardization
        self.exchange_mapping = {
            'NYSE/NASDAQ': ['NYSE', 'NASDAQ'],
            'NSE': 'NSE',
            'LSE': 'LSE',
            'TSX': 'TSX',
            'ASX': 'ASX'
        }
        
    def get_sp500_symbols(self) -> List[str]:
        """Fetch S&P 500 symbols from Wikipedia"""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]
            symbols = df['Symbol'].head(100).tolist()  # Get top 100 instead of 50
            logger.info(f"✅ Fetched {len(symbols)} S&P 500 symbols")
            return symbols
        except Exception as e:
            logger.error(f"❌ Error fetching S&P 500 symbols: {e}")
            # Fallback to popular S&P 500 stocks - expanded list
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
                'XOM', 'JPM', 'V', 'PG', 'HD', 'MA', 'CVX', 'ABBV', 'PFE', 'KO',
                'AVGO', 'PEP', 'COST', 'TMO', 'WMT', 'ABT', 'DHR', 'ACN', 'NEE', 'VZ',
                'TXN', 'ADBE', 'LIN', 'CRM', 'RTX', 'NKE', 'MRK', 'T', 'ORCL', 'CMCSA',
                'WFC', 'AMD', 'NFLX', 'COP', 'IBM', 'MDT', 'HON', 'UPS', 'QCOM', 'PM',
                'LOW', 'C', 'CAT', 'GE', 'INTC', 'SPGI', 'AXP', 'NOW', 'ISRG', 'INTU',
                'BKNG', 'TJX', 'GS', 'DE', 'SYK', 'ADP', 'MDLZ', 'VRTX', 'GILD', 'ADI',
                'BLK', 'MMM', 'CVS', 'CB', 'MO', 'SBUX', 'PYPL', 'TMUS', 'SO', 'ZTS',
                'CI', 'DUK', 'PLD', 'AMT', 'ANTM', 'ITW', 'EQIX', 'CL', 'BSX', 'FDX',
                'CSX', 'EOG', 'APD', 'CCI', 'NSC', 'WM', 'REGN', 'PGR', 'MMC', 'FCX'
            ]
    
    def get_index_symbols(self, index_key: str) -> List[str]:
        """Get symbols for a specific index"""
        config = self.indices_config[index_key]
        
        if index_key == 'SP500':
            return self.get_sp500_symbols()
        elif 'symbols' in config:
            return config['symbols'][:config['top_count']]
        else:
            logger.warning(f"No symbol source for {index_key}")
            return []
    
    def fetch_dividend_data(self, symbol: str, index_config: Dict) -> List[Dict]:
        """Fetch dividend data for a single symbol with quality filtering"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get basic info
            info = ticker.info
            company_name = info.get('longName', info.get('shortName', symbol))
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            market_cap = info.get('marketCap', 0)
            
            # Quality filter: Skip if market cap is too small (less than $1B)
            if market_cap and market_cap < 1_000_000_000:
                logger.info(f"⚠️  Skipping {symbol}: Market cap too small (${market_cap:,.0f})")
                return []
            
            # Get dividend history (last 10 years)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * 10)
            
            dividends = ticker.dividends
            if dividends.empty:
                logger.info(f"⚠️  No dividend data for {symbol}")
                return []
            
            # Quality filter: Must have at least 4 dividend payments in 10 years
            if len(dividends) < 4:
                logger.info(f"⚠️  Skipping {symbol}: Too few dividend payments ({len(dividends)})")
                return []
            
            # Filter to last 10 years - handle timezone aware dates
            try:
                # Always convert to naive datetime for comparison
                if hasattr(dividends.index, 'tz') and dividends.index.tz is not None:
                    dividends.index = dividends.index.tz_localize(None)
                dividends = dividends[dividends.index >= start_date]
            except Exception as e:
                logger.warning(f"⚠️ Timezone handling issue for {symbol}: {e}")
                # Fallback: try to convert index to naive
                try:
                    dividends.index = pd.to_datetime(dividends.index).tz_localize(None)
                    dividends = dividends[dividends.index >= start_date]
                except:
                    logger.error(f"❌ Could not process dates for {symbol}")
                    return []
            
            if dividends.empty:
                return []
            
            # Quality filter: Check for recent dividends (within last 2 years)
            recent_threshold = datetime.now() - timedelta(days=730)
            recent_dividends = dividends[dividends.index >= recent_threshold] if hasattr(dividends.index, 'tz') and dividends.index.tz else dividends[dividends.index >= recent_threshold]
            
            if recent_dividends.empty:
                logger.info(f"⚠️  Skipping {symbol}: No recent dividends (last 2 years)")
                return []
            
            # Get historical prices for dividend yield calculation
            hist = ticker.history(start=start_date, end=end_date)
            
            dividend_records = []
            for date, dividend_amount in dividends.items():
                try:
                    # Always convert to naive datetime for processing
                    if hasattr(date, 'tz') and date.tz is not None:
                        date_naive = date.tz_localize(None)
                    elif hasattr(date, 'tzinfo') and date.tzinfo is not None:
                        date_naive = date.replace(tzinfo=None)
                    else:
                        date_naive = date
                    
                    # Convert historical prices index to naive datetime
                    hist_index = hist.index
                    if hasattr(hist_index, 'tz') and hist_index.tz is not None:
                        hist_index = hist_index.tz_localize(None)
                    elif hasattr(hist_index, 'tzinfo'):
                        hist_index = pd.to_datetime(hist_index).tz_localize(None)
                    
                    # Find closest trading day for price
                    closest_dates = hist_index[hist_index <= date_naive]
                    if len(closest_dates) > 0:
                        price_date = closest_dates[-1]
                        # Get price using iloc for safer indexing
                        price_idx = list(hist_index).index(price_date)
                        share_price = hist.iloc[price_idx]['Close']
                        
                        dividend_yield = (dividend_amount / share_price) * 100 if share_price > 0 else 0
                        
                        # Determine exchange based on symbol suffix
                        exchange = self.determine_exchange(symbol, index_config)
                        
                        dividend_records.append({
                            'company_name': company_name,
                            'ticker_symbol': symbol,
                            'exchange': exchange,
                            'country': index_config['country'],
                            'currency': index_config['currency'],
                            'dividend_date': date_naive.strftime('%Y-%m-%d'),
                            'dividend_per_share': round(dividend_amount, 4),
                            'share_price_on_dividend_date': round(share_price, 2),
                            'dividend_yield_pct': round(dividend_yield, 2),
                            'sector': sector,
                            'industry': industry,
                            'market_cap': market_cap,
                            'last_updated': datetime.now().strftime('%Y-%m-%d'),
                            'index_source': index_config['name'],
                            'avg_dividend_yield': 0,  # Will calculate later
                            'exchange_rank': 0  # Will calculate later
                        })
                except Exception as e:
                    logger.warning(f"⚠️ Error processing dividend date {date} for {symbol}: {e}")
                    continue
            
            if dividend_records:
                # Quality filter: Calculate average yield and skip if too low
                avg_yield = sum(r['dividend_yield_pct'] for r in dividend_records) / len(dividend_records)
                if avg_yield < 0.5:  # Skip if average yield is less than 0.5%
                    logger.info(f"⚠️  Skipping {symbol}: Average yield too low ({avg_yield:.2f}%)")
                    return []
                
                logger.info(f"✅ {symbol}: {len(dividend_records)} dividend records (avg yield: {avg_yield:.2f}%)")
            
            return dividend_records
            
        except Exception as e:
            logger.error(f"❌ Error fetching {symbol}: {e}")
            return []
    
    def determine_exchange(self, symbol: str, index_config: Dict) -> str:
        """Determine exchange based on symbol suffix and index"""
        if symbol.endswith('.NS'):
            return 'NSE'
        elif symbol.endswith('.TO'):
            return 'TSX'
        elif symbol.endswith('.L'):
            return 'LSE'
        elif symbol.endswith('.AX'):
            return 'ASX'
        elif index_config['exchange'] == 'NYSE/NASDAQ':
            # For US stocks, try to determine NYSE vs NASDAQ
            # This is simplified - in reality you'd need additional lookup
            return 'NASDAQ' if len(symbol) <= 4 else 'NYSE'
        else:
            return index_config['exchange']
    
    def fetch_all_dividend_data(self, max_workers: int = 10) -> pd.DataFrame:
        """Fetch dividend data from all configured indices"""
        all_records = []
        
        for index_key, index_config in self.indices_config.items():
            logger.info(f"🔍 Processing {index_config['name']} ({index_config['country']})...")
            
            symbols = self.get_index_symbols(index_key)
            if not symbols:
                continue
            
            logger.info(f"📊 Fetching data for {len(symbols)} symbols from {index_config['name']}")
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_symbol = {
                    executor.submit(self.fetch_dividend_data, symbol, index_config): symbol 
                    for symbol in symbols
                }
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        records = future.result()
                        all_records.extend(records)
                    except Exception as e:
                        logger.error(f"❌ Failed to process {symbol}: {e}")
            
            # Add small delay between indices to be respectful to APIs
            time.sleep(2)
        
        if not all_records:
            logger.error("❌ No dividend data collected!")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_records)
        
        # Calculate average yields and rankings
        df = self.calculate_metrics(df)
        
        logger.info(f"✅ Collected {len(df)} dividend records from {len(df['ticker_symbol'].unique())} companies")
        logger.info(f"📈 Exchanges covered: {sorted(df['exchange'].unique())}")
        logger.info(f"🌍 Countries covered: {sorted(df['country'].unique())}")
        
        return df
    
    def calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate average dividend yields and exchange rankings"""
        if df.empty:
            return df
        
        # Calculate average dividend yield per company
        avg_yields = df.groupby('ticker_symbol')['dividend_yield_pct'].mean().round(2)
        df['avg_dividend_yield'] = df['ticker_symbol'].map(avg_yields)
        
        # Calculate exchange rankings based on average yield
        exchange_ranks = (df.groupby(['exchange', 'ticker_symbol'])['avg_dividend_yield']
                         .first()
                         .groupby('exchange')
                         .rank(method='dense', ascending=False))
        
        # Map back to main dataframe
        df['exchange_rank'] = df.apply(
            lambda row: exchange_ranks.get((row['exchange'], row['ticker_symbol']), 0), 
            axis=1
        )
        
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str = 'index_dividend_data.csv') -> str:
        """Save dividend data to CSV file"""
        if df.empty:
            logger.error("❌ No data to save!")
            return ""
        
        # Ensure data directory exists
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        filepath = os.path.join(data_dir, filename)
        
        # Sort by company name and dividend date
        df_sorted = df.sort_values(['company_name', 'dividend_date'])
        
        # Save to CSV
        df_sorted.to_csv(filepath, index=False)
        
        logger.info(f"💾 Saved {len(df_sorted)} records to {filepath}")
        logger.info(f"📊 Data summary:")
        logger.info(f"   • Companies: {len(df_sorted['ticker_symbol'].unique())}")
        logger.info(f"   • Exchanges: {len(df_sorted['exchange'].unique())}")
        logger.info(f"   • Date range: {df_sorted['dividend_date'].min()} to {df_sorted['dividend_date'].max()}")
        
        return filepath

def check_data_freshness(csv_path: str) -> bool:
    """Check if data needs updating (not from today)"""
    if not os.path.exists(csv_path):
        logger.info("📄 CSV file doesn't exist, will create new data")
        return False
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return False
        
        last_updated = df['last_updated'].iloc[0]
        today = datetime.now().strftime('%Y-%m-%d')
        
        if last_updated == today:
            logger.info(f"✅ Data is fresh (last updated: {last_updated})")
            return True
        else:
            logger.info(f"🔄 Data needs updating (last updated: {last_updated}, today: {today})")
            return False
    except Exception as e:
        logger.error(f"❌ Error checking data freshness: {e}")
        return False

def main():
    """Main function to update dividend data"""
    logger.info("🚀 Starting Index-Based Dividend Data Update")
    
    # Check if data needs updating
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'index_dividend_data.csv')
    
    if check_data_freshness(csv_path):
        logger.info("✅ Data is up to date, no update needed")
        return csv_path
    
    # Fetch fresh data
    fetcher = IndexBasedDividendFetcher()
    df = fetcher.fetch_all_dividend_data()
    
    if not df.empty:
        filepath = fetcher.save_data(df)
        logger.info("🎉 Data update completed successfully!")
        return filepath
    else:
        logger.error("❌ Failed to fetch dividend data")
        return ""

if __name__ == "__main__":
    main()
