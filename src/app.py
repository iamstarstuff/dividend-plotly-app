import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output
import pandas as pd
from data import load_data
from plots import create_bar_chart
from plotly.subplots import make_subplots
import plotly.graph_objs as go

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])  # Use a dark Bootstrap theme

# Load the dividend data
dividend_data = load_data()

if 'dividend_yield_pct' not in dividend_data.columns:
    dividend_data['dividend_yield_pct'] = (
        dividend_data['dividend_per_share'] / dividend_data['share_price_on_dividend_date']
    ) * 100

# Get top 50 companies by total dividend or frequency
top_by_total = dividend_data.groupby('company')['dividend_per_share'].sum().sort_values(ascending=False).head(50)
top_by_freq = dividend_data['company'].value_counts().head(50)
top_50_companies = pd.Index(top_by_total.index.tolist() + top_by_freq.index.tolist()).unique()

app.layout = html.Div([
    html.H1("Dividend Analysis Dashboard", style={'color': '#fff'}),
    # html.H5("Top 50 companies", style={'color': '#fff'}),
    dbc.Select(
        id='company-dropdown',
        options=[{'label': company, 'value': company} for company in top_50_companies],
        value=top_50_companies[0],
        style={'backgroundColor': '#222', 'color': '#fff', 'border': '1px solid #444'}
    ),
    html.P("Select a company from this dropdown.", style={'color': '#fff'}),
    dcc.Graph(
        id='total-dividend-bar-chart',
        config={'displayModeBar': False},
        style={'width': '100%', 'height': '80vh'}  # or '100vh' for full viewport height
    ),
    #dcc.Graph(id='yield-bar-chart', config={'displayModeBar': False})
], style={'backgroundColor': '#2222', 'minHeight': '100vh', 'padding': '20px'})

@app.callback(
    Output('total-dividend-bar-chart', 'figure'),
    Input('company-dropdown', 'value')
)
def update_graph(selected_company):
    filtered_data = dividend_data[dividend_data['company'] == selected_company]
    filtered_data = filtered_data.copy()
    filtered_data['dividend_date'] = pd.to_datetime(filtered_data['dividend_date'])
    filtered_data['year'] = filtered_data['dividend_date'].dt.year
    years = list(range(2015, 2026))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Bar for total dividend
    fig.add_bar(
        x=filtered_data['dividend_date'],
        y=filtered_data['dividend_per_share'],
        name='Dividend per Share',
        marker_color='#1f77b4',
        customdata=filtered_data[['share_price_on_dividend_date']],
        hovertemplate=(
            "Date: %{x}<br>"
            "Dividend: %{y:.2f}<br>"
            "Share Price: ₹%{customdata[0]:.2f}<extra></extra>"
        ),
    )
    # Line for dividend yield
    fig.add_trace(
        go.Scatter(
            x=filtered_data['dividend_date'],
            y=filtered_data['dividend_yield_pct'],
            name='Dividend Yield (%)',
            mode='lines+markers',
            marker=dict(color='#ff7f0e'),
            yaxis='y2',
            hovertemplate=(
                "Date: %{x}<br>"
                "Yield: %{y:.2f}%<extra></extra>"
            ),
        ),
        secondary_y=True
    )

    fig.update_layout(
        title=f"Dividends & Yield Over Time: {selected_company}",
        plot_bgcolor='#222',
        paper_bgcolor='#222',
        font_color='#fff',
        xaxis=dict(
            color='#fff',
            title='Dividend Date',
            tickmode='array',
            tickvals=years,
            ticktext=[str(y) for y in years],
            tickfont=dict(color='#fff'),
            gridcolor='#444',
            gridwidth=1,
            #type='category'
        ),
        yaxis=dict(
            color='#fff',
            title='Dividend per Share',
            tickfont=dict(color='#fff'),
            gridwidth=1
        ),
        yaxis2=dict(
            color='#fff',
            title='Dividend Yield (%)',
            tickfont=dict(color='#fff'),
            gridwidth=1,
            griddash='dash',
            gridcolor='#ff7f0e'
        ),
        legend=dict(font=dict(color='#fff'))
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)