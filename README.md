# Global High-Dividend Stocks Dashboard

A scalable web application built with Plotly Dash to analyze and visualize dividend data from worldwide stock exchanges. This app focuses on stocks with high dividend yields (>10%) and provides automated data updates.

## 🌟 Features

- **Global Coverage**: Supports major stock exchanges worldwide (NYSE, NASDAQ, LSE, TSE, NSE, ASX, etc.)
- **High-Dividend Focus**: Filters and displays only stocks with >10% dividend yield
- **Real-time Updates**: Automated daily data fetching and updates
- **Interactive Filtering**: Filter by exchange, country, and dividend yield
- **Scalable Architecture**: Database-backed with CSV fallback
- **Multi-source Data**: Integrates Yahoo Finance, Alpha Vantage, and other APIs

## 🏗️ Architecture

### Scalable Design
```
├── src/
│   ├── app.py                 # Main Dash application
│   ├── config/
│   │   ├── config.py          # Global configuration
│   │   └── __init__.py
│   ├── database/
│   │   ├── models.py          # SQLAlchemy models
│   │   ├── init_db.py         # Database initialization
│   │   └── __init__.py
│   ├── data_fetching/
│   │   ├── fetchers.py        # Multi-source data fetching
│   │   └── __init__.py
│   ├── update_scripts/
│   │   ├── daily_updater.py   # Automated update script
│   │   └── __init__.py
│   ├── data/
│   │   ├── __init__.py        # Enhanced data loading
│   │   └── *.csv              # CSV fallback files
│   └── plots/
│       └── __init__.py        # Plotting utilities
├── requirements.txt           # Dependencies
├── deploy.sh                 # Deployment script
├── .env.example              # Environment configuration
└── README.md
```

### Database Schema
- **dividend_data**: Main table storing dividend information
- **exchange_info**: Exchange metadata and trading hours
- **data_sources**: API source reliability tracking

## 🚀 Setup Instructions

### 1. Clone and Install
```bash
git clone <repository-url>
cd dividend-plotly-app
pip install -r requirements.txt
```

### 2. Environment Configuration
```bash
cp .env.example .env
# Edit .env with your database and API credentials
```

### 3. Database Setup
```bash
# Option A: Use PostgreSQL (Recommended for production)
export DATABASE_URL="postgresql://user:password@localhost:5432/dividend_db"

# Option B: Use SQLite (For development)
export DATABASE_URL="sqlite:///dividend_data.db"

# Initialize database
python src/database/init_db.py
```

### 4. API Keys (Optional but Recommended)
```bash
# Get free API key from Alpha Vantage
export ALPHA_VANTAGE_API_KEY="your_api_key_here"
```

### 5. Initial Data Load
```bash
python src/update_scripts/daily_updater.py --once
```

### 6. Run the Application
```bash
python src/app.py
```

## 📊 Data Sources & Scaling Strategy

### Current Data Sources
1. **Yahoo Finance** (Primary, Free)
   - Global coverage
   - Real-time data
   - Rate limit: ~2000 requests/hour

2. **Alpha Vantage** (Secondary, API Key Required)
   - Professional data quality
   - Rate limit: 5 calls/minute (free tier)

### Scaling to Worldwide Coverage

#### Phase 1: Enhanced Data Fetching (Current)
- Multi-threaded data fetching
- Rate limiting and retry logic
- Multiple API source integration
- Automatic fallback between sources

#### Phase 2: Exchange-Specific APIs
```python
# Add specialized fetchers for each exchange
class NSEFetcher(DataFetcher):
    def fetch_exchange_tickers(self):
        # NSE-specific API calls
        pass

class LSEFetcher(DataFetcher):
    def fetch_exchange_tickers(self):
        # LSE-specific API calls
        pass
```

#### Phase 3: Professional Data Providers
- Bloomberg Terminal API
- Refinitiv (Reuters) API
- Quandl integration
- Interactive Brokers API

### Automated Updates

#### Daily Update Process
1. **Data Fetching**: Multi-threaded retrieval from all sources
2. **Data Validation**: Check for anomalies and duplicates
3. **Database Update**: Upsert new dividend records
4. **CSV Export**: Backward compatibility with CSV format
5. **Cleanup**: Remove old records (>1 year)

#### Scheduling Options
```bash
# Option A: Cron job (Unix/Linux)
0 6 * * * cd /path/to/app && python src/update_scripts/daily_updater.py

# Option B: Python scheduler (Cross-platform)
python src/update_scripts/daily_updater.py  # Runs continuously

# Option C: Docker with scheduler
docker run -d --name dividend-updater dividend-app:latest
```

## 🔧 Configuration

### Key Settings in `config/config.py`
```python
@dataclass
class AppConfig:
    min_dividend_yield: float = 10.0        # Minimum yield threshold
    max_companies_per_exchange: int = 1000  # Limit per exchange
    update_frequency_hours: int = 24        # Update frequency
    supported_exchanges: List[str] = [       # Target exchanges
        'NYSE', 'NASDAQ', 'LSE', 'TSE', 'NSE', 'ASX', 'FRA', 'TSX'
    ]
```

### Environment Variables
```bash
# Database
DB_HOST=localhost
DB_NAME=dividend_db
DB_USER=postgres
DB_PASSWORD=your_password

# APIs
ALPHA_VANTAGE_API_KEY=your_key
QUANDL_API_KEY=your_key

# App Settings
MIN_DIVIDEND_YIELD=10.0
UPDATE_FREQUENCY_HOURS=24
```

## 📈 Performance Optimizations

### Database Optimizations
- Indexed queries on ticker, exchange, dividend_yield
- Partitioning by date ranges
- Connection pooling
- Bulk insert operations

### Data Fetching Optimizations
- Concurrent API calls with rate limiting
- Intelligent retry logic with exponential backoff
- Data source reliability scoring
- Caching of exchange metadata

### Application Optimizations
- Data loader caching
- Efficient DataFrame operations
- Responsive UI with loading states
- Progressive data loading

## 🚀 Deployment

### Quick Deployment
```bash
chmod +x deploy.sh
./deploy.sh
```

### Production Deployment
```bash
# Use PostgreSQL
export DATABASE_URL="postgresql://user:pass@host:5432/db"

# Use process manager
pip install gunicorn
gunicorn src.app:server -b 0.0.0.0:8050 --workers 4

# Or use Docker
docker build -t dividend-app .
docker run -p 8050:8050 dividend-app
```

### Cloud Deployment Options
- **Heroku**: Easy deployment with Postgres addon
- **AWS**: EC2 + RDS PostgreSQL
- **Google Cloud**: App Engine + Cloud SQL
- **DigitalOcean**: Droplet + Managed Database

## 🔍 Monitoring & Maintenance

### Logging
- All data fetching operations logged
- Error tracking and alerting
- Performance metrics collection

### Data Quality Checks
- Dividend yield validation (reasonable ranges)
- Price anomaly detection
- Duplicate record prevention
- Missing data alerts

### Maintenance Tasks
```bash
# Check data freshness
python src/update_scripts/daily_updater.py --check-health

# Manual data refresh
python src/update_scripts/daily_updater.py --once

# Database cleanup
python src/update_scripts/daily_updater.py --cleanup
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.