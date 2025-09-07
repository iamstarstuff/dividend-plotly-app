"""
Enhanced data loading for comprehensive dividend data
"""
import pandas as pd
import os
from typing import List, Dict
import logging
from datetime import datetime
import subprocess
import sys

logger = logging.getLogger(__name__)

class DataLoader:
    """Enhanced data loader for comprehensive dividend database"""
    
    def __init__(self):
        self.data_dir = os.path.dirname(__file__)
        self.comprehensive_csv = os.path.join(self.data_dir, 'comprehensive_dividend_data.csv')
        self.high_dividend_csv = os.path.join(self.data_dir, 'focused_high_yield_dividends_20250906_215020.csv')
        self.legacy_csv = os.path.join(self.data_dir, 'index_dividend_data.csv')
        self._cached_data = None
        self._cache_timestamp = None
        
        # Build comprehensive database if it doesn't exist
        self._ensure_comprehensive_database()
        
    def _ensure_comprehensive_database(self):
        """Ensure comprehensive database exists, build if needed"""
        if not os.path.exists(self.comprehensive_csv):
            logger.info("🏗️ Comprehensive database not found. Building...")
            try:
                builder_script = os.path.join(
                    os.path.dirname(self.data_dir), 
                    'update_scripts', 
                    'comprehensive_dividend_builder.py'
                )
                
                if os.path.exists(builder_script):
                    logger.info("📦 Running comprehensive dividend builder...")
                    result = subprocess.run([sys.executable, builder_script], 
                                          capture_output=True, text=True, timeout=1800)  # 30 min timeout
                    
                    if result.returncode == 0:
                        logger.info("✅ Comprehensive database built successfully")
                    else:
                        logger.error(f"❌ Database build failed: {result.stderr}")
                        
            except Exception as e:
                logger.error(f"Error building comprehensive database: {e}")
    
    def load_data(self, min_dividend_yield: float = 10.0, force_refresh: bool = False) -> pd.DataFrame:
        """Load comprehensive dividend data with caching"""
        # Check cache
        if not force_refresh and self._cached_data is not None and self._cache_timestamp:
            cache_age = (datetime.now() - self._cache_timestamp).total_seconds() / 3600
            if cache_age < 1:  # Cache for 1 hour
                logger.info("Using cached data")
                return self._apply_filters(self._cached_data, min_dividend_yield)
        
        # Load fresh data
        df = self._load_from_csv()
        
        if not df.empty:
            self._cached_data = df
            self._cache_timestamp = datetime.now()
            logger.info(f"Loaded {len(df)} records, cached at {self._cache_timestamp}")
        
        return self._apply_filters(df, min_dividend_yield)
    
    def _load_from_csv(self) -> pd.DataFrame:
        """Load data from available CSV files (prioritize comprehensive data)"""
        # Try comprehensive database first
        if os.path.exists(self.comprehensive_csv):
            try:
                df = pd.read_csv(self.comprehensive_csv)
                logger.info(f"Loaded {len(df)} records from comprehensive database")
                return self._process_dataframe(df)
            except Exception as e:
                logger.error(f"Error loading comprehensive database: {e}")
        
        # Try high dividend stocks file
        if os.path.exists(self.high_dividend_csv):
            try:
                df = pd.read_csv(self.high_dividend_csv)
                logger.info(f"✅ Loading HIGH-YIELD dividend data (global discovery)")
                return self._process_dataframe(df)
            except Exception as e:
                logger.error(f"Error loading high-yield data: {e}")
        
        # Try legacy file
        if os.path.exists(self.legacy_csv):
            try:
                df = pd.read_csv(self.legacy_csv)
                logger.info(f"📊 Loading INDEX dividend data (fallback)")
                return self._process_dataframe(df)
            except Exception as e:
                logger.error(f"Error loading index data: {e}")
        
        # Return empty DataFrame if all fails
        logger.warning("No data sources available, returning empty DataFrame")
        return self._create_empty_dataframe()
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean DataFrame"""
        # Convert date columns
        if 'dividend_date' in df.columns:
            df['dividend_date'] = pd.to_datetime(df['dividend_date'], errors='coerce')
        
        # Calculate dividend yield if missing
        if 'dividend_yield_pct' not in df.columns:
            if 'dividend_per_share' in df.columns and 'share_price_on_dividend_date' in df.columns:
                # Avoid division by zero
                mask = df['share_price_on_dividend_date'] > 0
                df.loc[mask, 'dividend_yield_pct'] = (
                    df.loc[mask, 'dividend_per_share'] / df.loc[mask, 'share_price_on_dividend_date'] * 100
                )
        
        # Add company column for backward compatibility
        if 'company' not in df.columns and 'company_name' in df.columns:
            df['company'] = df['company_name']
        elif 'company_name' not in df.columns and 'company' in df.columns:
            df['company_name'] = df['company']
        
        # Add default values for missing columns
        if 'exchange' not in df.columns:
            df['exchange'] = 'UNKNOWN'
        if 'country' not in df.columns:
            df['country'] = 'UNK'
        if 'currency' not in df.columns:
            df['currency'] = 'USD'
        
        return df
    
    def _process_legacy_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process legacy DataFrame format"""
        # Rename columns for consistency
        column_mapping = {
            'Company': 'company_name',
            'Symbol': 'ticker_symbol', 
            'Dividend Date': 'dividend_date',
            'Dividend': 'dividend_per_share',
            'Price': 'share_price_on_dividend_date'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Add missing columns with defaults
        if 'exchange' not in df.columns:
            df['exchange'] = 'NSE'  # Assume NSE for legacy data
        if 'country' not in df.columns:
            df['country'] = 'IND'
        if 'currency' not in df.columns:
            df['currency'] = 'INR'
        
        return self._process_dataframe(df)
    
    def _apply_filters(self, df: pd.DataFrame, min_dividend_yield: float) -> pd.DataFrame:
        """Apply filters to the data"""
        if df.empty:
            return df
        
        # Filter for minimum dividend yield
        if 'dividend_yield_pct' in df.columns:
            df = df[df['dividend_yield_pct'] >= min_dividend_yield]
        
        return df
    
    def _create_empty_dataframe(self) -> pd.DataFrame:
        """Create empty DataFrame with required columns"""
        return pd.DataFrame({
            'company': ['No Data Available'],
            'company_name': ['No Data Available'],
            'ticker_symbol': ['N/A'],
            'dividend_date': [pd.Timestamp.now()],
            'dividend_per_share': [0],
            'share_price_on_dividend_date': [1],  # Avoid division by zero
            'dividend_yield_pct': [0],
            'exchange': ['N/A'],
            'country': ['N/A'],
            'currency': ['USD']
        })
    
    def get_all_companies(self) -> List[Dict]:
        """Get all companies with their metadata for dropdown"""
        df = self.load_data(min_dividend_yield=0)  # Get all data
        if df.empty:
            return [{'label': 'No Data Available', 'value': 'none'}]
        
        # Group by company and calculate stats
        company_stats = df.groupby(['ticker_symbol', 'company_name', 'exchange', 'country']).agg({
            'dividend_yield_pct': ['mean', 'count'],
            'dividend_date': ['min', 'max']
        }).round(2)
        
        company_stats.columns = ['avg_yield', 'dividend_count', 'first_dividend', 'last_dividend']
        company_stats = company_stats.reset_index()
        
        # Filter for companies with sufficient dividend data (at least 5 records)
        companies_with_data = company_stats[company_stats['dividend_count'] >= 5]
        
        # Sort by average yield descending
        companies_with_data = companies_with_data.sort_values('avg_yield', ascending=False)
        
        # Create dropdown options
        options = []
        for _, row in companies_with_data.iterrows():
            label = f"{row['ticker_symbol']} - {row['company_name']} ({row['exchange']}) - Avg: {row['avg_yield']:.1f}%"
            options.append({
                'label': label,
                'value': row['ticker_symbol']
            })
        
        return options
    
    def get_company_data(self, ticker_symbol: str) -> pd.DataFrame:
        """Get all dividend data for a specific company"""
        df = self.load_data(min_dividend_yield=0)  # Get all data
        if df.empty:
            return df
            
        company_data = df[df['ticker_symbol'] == ticker_symbol].copy()
        
        if not company_data.empty:
            # Sort by dividend date
            company_data = company_data.sort_values('dividend_date')
            
            # Add year column for grouping
            company_data['year'] = company_data['dividend_date'].dt.year
        
        return company_data
    
    def get_exchanges(self) -> List[str]:
        """Get list of available exchanges"""
        df = self.load_data()
        if 'exchange' in df.columns:
            return sorted(df['exchange'].unique().tolist())
        return ['ALL']
    
    def get_countries(self) -> List[str]:
        """Get list of available countries"""
        df = self.load_data()
        if 'country' in df.columns:
            return sorted(df['country'].unique().tolist())
        return ['ALL']
    
    def save_data(self, df: pd.DataFrame, filename: str = None):
        """Save DataFrame to CSV"""
        if filename is None:
            filename = self.high_dividend_csv
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Save with timestamp
            df.to_csv(filename, index=False)
            logger.info(f"Saved {len(df)} records to {filename}")
            
            # Update cache
            self._cached_data = df
            self._cache_timestamp = datetime.now()
            
        except Exception as e:
            logger.error(f"Error saving data to {filename}: {e}")

# Global data loader instance
_data_loader = None

def get_data_loader() -> DataLoader:
    """Get singleton data loader instance"""
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader()
    return _data_loader

def load_data() -> pd.DataFrame:
    """Legacy function for backward compatibility"""
    return get_data_loader().load_data()