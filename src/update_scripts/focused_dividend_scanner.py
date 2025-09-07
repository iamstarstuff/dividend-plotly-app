"""
Focused High-Yield Dividend Stock Scanner
Targets specific high-yield categories with better rate limiting
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Dict, List, Tuple, Optional
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FocusedDividendScanner:
    """Focused scanner for known high-yield dividend categories"""
    
    def __init__(self):
        # Curated list of known high-yield dividend stocks
        self.high_yield_stocks = {
            'mortgage_reits': {
                'description': 'Mortgage REITs (typically 8-25% yields)',
                'symbols': [
                    'ORC', 'AGNC', 'NYMT', 'NLY', 'CIM', 'TWO', 'ARR', 'IVR',
                    'PMT', 'REML', 'AI', 'DX', 'MITT', 'MFA', 'CHMI', 'NRZ'
                ]
            },
            'bdc_companies': {
                'description': 'Business Development Companies (typically 8-15% yields)',
                'symbols': [
                    'PSEC', 'MAIN', 'ARCC', 'HTGC', 'GAIN', 'NEWT', 'TSLX',
                    'PTMN', 'GLAD', 'TCPC', 'FDUS', 'PFLT', 'CGBD', 'CSWC',
                    'GBDC', 'SLRC', 'OCSL', 'ORCC', 'NMFC', 'GSBD'
                ]
            },
            'closed_end_funds': {
                'description': 'High-Yield Closed-End Funds (typically 6-20% yields)',
                'symbols': [
                    'ECC', 'EIM', 'EOS', 'EOI', 'ETO', 'EVT', 'EXG', 'ETG',
                    'JPC', 'JPS', 'JPT', 'JPI', 'JHS', 'JHI', 'JQC', 'JFR',
                    'NCV', 'NCZ', 'NUV', 'NXJ', 'NXP', 'NXR', 'NZF', 'NAZ'
                ]
            },
            'energy_trusts': {
                'description': 'Canadian Energy Income Trusts (typically 5-15% yields)',
                'symbols': [
                    'PEY.TO', 'AAV.TO', 'BIR.TO', 'CR.TO', 'ERF.TO', 'GXE.TO',
                    'KEL.TO', 'NVA.TO', 'OBE.TO', 'POU.TO', 'SGY.TO', 'TOU.TO',
                    'VET.TO', 'WCP.TO', 'YGR.TO', 'PIPE.TO', 'FRU.TO', 'CJ.TO'
                ]
            },
            'utility_reits': {
                'description': 'Utility and Infrastructure REITs (typically 4-12% yields)',
                'symbols': [
                    'O', 'STAG', 'WPC', 'NNN', 'STORE', 'ADC', 'SRC', 'NETL',
                    'AMT', 'CCI', 'SBAC', 'EQIX', 'DLR', 'PLD', 'EXR', 'PSA'
                ]
            },
            'preferred_stocks': {
                'description': 'Bank Preferred Stocks (typically 4-8% yields)',
                'symbols': [
                    'BAC-PL', 'BAC-PK', 'C-PJ', 'C-PN', 'WFC-PL', 'WFC-PQ',
                    'GS-PJ', 'GS-PK', 'MS-PA', 'MS-PE', 'JPM-PC', 'JPM-PD'
                ]
            },
            'tobacco_telecom': {
                'description': 'High-Yield Tobacco & Telecom (typically 4-10% yields)',
                'symbols': [
                    'BTI', 'PM', 'MO', 'T', 'VZ', 'VOD', 'BCE', 'TU',
                    'TEF', 'VIV', 'ORAN', 'NTT', 'KT', 'CHT'
                ]
            }
        }
    
    def fetch_comprehensive_dividend_data(self, symbol: str) -> List[Dict]:
        """Fetch comprehensive dividend data for a single stock"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get basic stock info
            info = ticker.info
            if not info or 'symbol' not in info:
                logger.debug(f"⚠️ No info data for {symbol}")
                return []
            
            company_name = info.get('longName', info.get('shortName', symbol))
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            market_cap = info.get('marketCap', 0)
            exchange = info.get('exchange', 'Unknown')
            currency = info.get('currency', 'USD')
            country = info.get('country', 'Unknown')
            
            # Get multiple yield measures
            dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            trailing_yield = info.get('trailingAnnualDividendYield', 0) * 100 if info.get('trailingAnnualDividendYield') else 0
            forward_yield = info.get('forwardAnnualDividendYield', 0) * 100 if info.get('forwardAnnualDividendYield') else 0
            
            # Get dividend history (last 5 years)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * 5)
            
            dividends = ticker.dividends
            if dividends.empty:
                logger.debug(f"⚠️ No dividend history for {symbol}")
                return []
            
            # Handle timezone issues
            try:
                if hasattr(dividends.index, 'tz') and dividends.index.tz is not None:
                    dividends.index = dividends.index.tz_localize(None)
                dividends = dividends[dividends.index >= start_date]
            except Exception as e:
                logger.debug(f"⚠️ Date filtering issue for {symbol}: {e}")
                # Fallback: take recent dividends
                dividends = dividends.tail(20)
            
            if dividends.empty:
                logger.debug(f"⚠️ No recent dividends for {symbol}")
                return []
            
            # Get historical prices
            hist = ticker.history(start=start_date, end=end_date)
            if hist.empty:
                logger.debug(f"⚠️ No price history for {symbol}")
                return []
            
            # Handle timezone for historical data
            try:
                if hasattr(hist.index, 'tz') and hist.index.tz is not None:
                    hist.index = hist.index.tz_localize(None)
            except:
                pass
            
            # Calculate current yield (use the highest available)
            current_yield = max(dividend_yield, trailing_yield, forward_yield)
            
            # Process each dividend payment
            dividend_records = []
            for date, dividend_amount in dividends.items():
                try:
                    # Convert timezone-aware date to naive
                    if hasattr(date, 'tz') and date.tz is not None:
                        date_naive = date.tz_localize(None)
                    elif hasattr(date, 'tzinfo') and date.tzinfo is not None:
                        date_naive = date.replace(tzinfo=None)
                    else:
                        date_naive = date
                    
                    # Find closest price
                    hist_index = hist.index
                    closest_dates = hist_index[hist_index <= date_naive]
                    if len(closest_dates) > 0:
                        price_date = closest_dates[-1]
                        price_idx = list(hist_index).index(price_date)
                        share_price = hist.iloc[price_idx]['Close']
                        
                        # Calculate yield at that time
                        calculated_yield = (dividend_amount / share_price) * 100 if share_price > 0 else 0
                        
                        dividend_records.append({
                            'company_name': company_name,
                            'ticker_symbol': symbol,
                            'exchange': exchange,
                            'country': country,
                            'currency': currency,
                            'dividend_date': date_naive.strftime('%Y-%m-%d'),
                            'dividend_per_share': round(dividend_amount, 4),
                            'share_price_on_dividend_date': round(share_price, 2),
                            'dividend_yield_pct': round(calculated_yield, 2),
                            'sector': sector,
                            'industry': industry,
                            'market_cap': market_cap,
                            'last_updated': datetime.now().strftime('%Y-%m-%d'),
                            'current_yield': round(current_yield, 2),
                            'trailing_yield': round(trailing_yield, 2),
                            'forward_yield': round(forward_yield, 2),
                            'discovery_method': 'focused_scan'
                        })
                except Exception as e:
                    logger.debug(f"⚠️ Error processing dividend date for {symbol}: {e}")
                    continue
            
            if dividend_records:
                avg_yield = sum(r['dividend_yield_pct'] for r in dividend_records) / len(dividend_records)
                logger.info(f"✅ {symbol}: {len(dividend_records)} records, current: {current_yield:.2f}%, avg: {avg_yield:.2f}%")
                
                # Add rate limiting
                time.sleep(0.2)  # 200ms delay between requests
            
            return dividend_records
            
        except Exception as e:
            logger.error(f"❌ Error fetching {symbol}: {e}")
            time.sleep(1)  # Longer delay on error
            return []
    
    def scan_high_yield_categories(self, max_workers: int = 10) -> pd.DataFrame:
        """Scan all high-yield categories with controlled rate limiting"""
        all_symbols = []
        category_map = {}
        
        # Collect all symbols and track their categories
        for category, data in self.high_yield_stocks.items():
            symbols = data['symbols']
            all_symbols.extend(symbols)
            for symbol in symbols:
                category_map[symbol] = category
            logger.info(f"📊 Added {len(symbols)} symbols from {data['description']}")
        
        logger.info(f"🔍 Total symbols to scan: {len(all_symbols)}")
        
        all_records = []
        processed_count = 0
        
        # Process in smaller batches with rate limiting
        batch_size = 20
        for i in range(0, len(all_symbols), batch_size):
            batch = all_symbols[i:i+batch_size]
            logger.info(f"📊 Processing batch {i//batch_size + 1}: symbols {i+1} to {min(i+batch_size, len(all_symbols))}")
            
            with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as executor:
                future_to_symbol = {
                    executor.submit(self.fetch_comprehensive_dividend_data, symbol): symbol 
                    for symbol in batch
                }
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        records = future.result()
                        if records:
                            # Add category information
                            category = category_map.get(symbol, 'unknown')
                            for record in records:
                                record['dividend_category'] = category
                            all_records.extend(records)
                        
                        processed_count += 1
                        if processed_count % 10 == 0:
                            logger.info(f"📈 Processed {processed_count}/{len(all_symbols)} symbols...")
                            
                    except Exception as e:
                        logger.error(f"❌ Failed to process {symbol}: {e}")
            
            # Batch delay to avoid rate limiting
            if i + batch_size < len(all_symbols):
                logger.info("⏳ Pausing between batches...")
                time.sleep(2)
        
        if not all_records:
            logger.error("❌ No dividend data collected!")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_records)
        
        # Calculate additional metrics
        df = self.calculate_enhanced_metrics(df)
        
        # Sort by current yield descending
        df = df.sort_values(['current_yield', 'ticker_symbol'], ascending=[False, True])
        
        logger.info(f"✅ Collected {len(df)} dividend records from {len(df['ticker_symbol'].unique())} companies")
        logger.info(f"📈 Yield range: {df['current_yield'].min():.2f}% - {df['current_yield'].max():.2f}%")
        
        return df
    
    def calculate_enhanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced dividend metrics"""
        if df.empty:
            return df
        
        # Calculate per-company metrics
        company_stats = df.groupby('ticker_symbol').agg({
            'dividend_yield_pct': ['mean', 'std', 'count'],
            'dividend_per_share': ['sum', 'mean'],
            'current_yield': 'first'
        }).round(2)
        
        # Flatten column names
        company_stats.columns = ['avg_yield', 'yield_volatility', 'payment_count', 
                               'total_dividends', 'avg_dividend_amount', 'current_yield_ref']
        
        # Map back to original DataFrame
        df['avg_dividend_yield'] = df['ticker_symbol'].map(company_stats['avg_yield'])
        df['yield_volatility'] = df['ticker_symbol'].map(company_stats['yield_volatility'])
        df['payment_frequency'] = df['ticker_symbol'].map(company_stats['payment_count'])
        df['annual_dividend_estimate'] = df['ticker_symbol'].map(company_stats['total_dividends'])
        
        # Calculate yield consistency score (lower volatility = higher score)
        max_volatility = df['yield_volatility'].max()
        df['yield_consistency_score'] = (max_volatility - df['yield_volatility']) / max_volatility * 100
        df['yield_consistency_score'] = df['yield_consistency_score'].fillna(100).round(1)
        
        # Calculate rankings within categories
        df['category_yield_rank'] = df.groupby('dividend_category')['current_yield'].rank(method='dense', ascending=False)
        
        return df
    
    def save_focused_data(self, df: pd.DataFrame) -> str:
        """Save focused high-yield dividend data"""
        if df.empty:
            logger.error("❌ No data to save!")
            return ""
        
        # Ensure data directory exists
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'focused_high_yield_dividends_{timestamp}.csv'
        filepath = os.path.join(data_dir, filename)
        
        # Save full data
        df.to_csv(filepath, index=False)
        
        # Create summary by company
        company_summary = df.groupby('ticker_symbol').agg({
            'company_name': 'first',
            'current_yield': 'first',
            'avg_dividend_yield': 'first',
            'payment_frequency': 'first',
            'yield_volatility': 'first',
            'dividend_category': 'first',
            'country': 'first',
            'sector': 'first',
            'market_cap': 'first'
        }).round(2)
        
        company_summary = company_summary.sort_values('current_yield', ascending=False)
        
        # Save company summary
        summary_filepath = filepath.replace('.csv', '_companies.csv')
        company_summary.to_csv(summary_filepath)
        
        # Print top performers by category
        logger.info(f"\n💾 Saved {len(df)} records to {filepath}")
        logger.info(f"📊 Company summary saved to {summary_filepath}")
        
        # Show top performers by category
        for category in df['dividend_category'].unique():
            cat_data = company_summary[company_summary['dividend_category'] == category]
            if not cat_data.empty:
                top_5 = cat_data.head(5)
                logger.info(f"\n🏆 TOP 5 in {category.upper()}:")
                for ticker, row in top_5.iterrows():
                    logger.info(f"   {ticker:<12} {row['current_yield']:>6.2f}% {row['company_name']}")
        
        # Overall top 20
        logger.info(f"\n🥇 TOP 20 OVERALL DIVIDEND YIELDS:")
        top_20 = company_summary.head(20)
        for ticker, row in top_20.iterrows():
            logger.info(f"   {ticker:<12} {row['current_yield']:>6.2f}% ({row['dividend_category']}) {row['company_name']}")
        
        return filepath

def main():
    """Main function to run focused dividend scanning"""
    logger.info("🎯 Starting Focused High-Yield Dividend Scanner")
    
    scanner = FocusedDividendScanner()
    
    # Run the scan with controlled rate limiting
    df = scanner.scan_high_yield_categories(max_workers=8)
    
    if not df.empty:
        filepath = scanner.save_focused_data(df)
        logger.info(f"🎉 Focused dividend scan completed!")
        logger.info(f"📄 Data saved to: {filepath}")
        return filepath
    else:
        logger.error("❌ Failed to collect dividend data")
        return ""

if __name__ == "__main__":
    main()
