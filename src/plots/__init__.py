import plotly.graph_objs as go

def create_bar_chart(df, x_column, y_column, title):
    fig = go.Figure()
    fig.add_bar(
        x=df[x_column],
        y=df[y_column],
        customdata=df[['share_price_on_dividend_date']],
        marker_color='#1f77b4',
        hovertemplate=(
            x_column + ": %{x}<br>"
            + y_column + ": %{y:.2f}<br>"
            + "Share Price: ₹%{customdata[0]:.2f}<extra></extra>"
        ),
    )
    fig.update_layout(
        title=title,
        plot_bgcolor='#222',
        paper_bgcolor='#222',
        font_color='#fff',
        xaxis=dict(
            color='#fff',
            title=dict(text=x_column, font=dict(color='#fff')),
            tickfont=dict(color='#fff')
        ),
        yaxis=dict(
            color='#fff',
            title=dict(text=y_column, font=dict(color='#fff')),
            tickfont=dict(color='#fff')
        ),
    )
    return fig

def create_scatter_plot(data, x_column, y_column, title):
    import plotly.express as px
    fig = px.scatter(data, x=x_column, y=y_column, title=title)
    return fig

def create_line_chart(data, x_column, y_column, title):
    import plotly.express as px
    fig = px.line(data, x=x_column, y=y_column, title=title)
    return fig

def create_pie_chart(data, names_column, values_column, title):
    import plotly.express as px
    fig = px.pie(data, names=names_column, values=values_column, title=title)
    return fig