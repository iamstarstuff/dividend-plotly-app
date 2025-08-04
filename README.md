# Dividend Analysis with Plotly Dash

This project is a web application built using Plotly Dash to visualize and analyze dividend data. It allows users to interactively filter and explore dividend information from various companies.

## Project Structure

```
dividend-plotly-app
├── src
│   ├── app.py          # Main entry point of the application
│   ├── data
│   │   └── __init__.py # Data loading and preprocessing functions
│   ├── plots
│   │   └── __init__.py # Functions for creating Plotly graphs
│   └── utils
│       └── __init__.py # Utility functions for data manipulation
├── requirements.txt     # List of dependencies
└── README.md            # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd dividend-plotly-app
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment. You can create one using:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

   Then install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. **Run the application**:
   Execute the following command to start the Dash application:
   ```
   python src/app.py
   ```

4. **Access the application**:
   Open your web browser and go to `http://127.0.0.1:8050` to view the application.

## Features

- **Interactive Filtering**: Users can filter dividend data based on various criteria such as company name, dividend yield, and frequency of dividends.
- **Dynamic Visualizations**: The application provides various visualizations including bar charts and scatter plots to analyze dividend trends.
- **Data Insights**: Gain insights into the performance of different companies based on their dividend payouts over time.

## Usage Examples

- Select a company from the dropdown to view its dividend history.
- Adjust the filters to see how dividend yields compare across different companies.
- Explore the scatter plots to understand the relationship between dividend yield and frequency.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.