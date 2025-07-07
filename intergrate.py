import pandas as pd
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

# ─── 1) Load Data ────────────────────────────────────────────────────────────────
df = pd.read_csv('dataset/preprocess_dataset2.csv')

# ─── 2) Compute KPIs ─────────────────────────────────────────────────────────────
total_jobs      = len(df)
avg_min_salary  = df['min_salary'].mean()
avg_mean_salary = df['mean_salary'].mean()

# ─── 3) Prepare Aggregation for Bar Chart ────────────────────────────────────────
job_counts = (
    df['category']
      .value_counts()
      .reset_index()
)
job_counts.columns = ['Category', 'Count']

# ─── 4) Shared Plotly template ───────────────────────────────────────────────────
PX_TEMPLATE = 'plotly_white'

# ─── 5) Initialize App ──────────────────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="JobStreet Interactive Dashboard"
)
server = app.server

# ─── 6) Navbar & KPI Cards ──────────────────────────────────────────────────────
navbar = dbc.NavbarSimple(
    brand="📈 JobStreet Insights",
    color="dark",
    dark=True,
    fluid=True,
)

kpi_row = dbc.Row([
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.H6("Total Postings", className="card-title"),
            html.H2(f"{total_jobs:,}", className="card-text")
        ])
    ], color="info", inverse=True), width=4),
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.H6("Avg. Min Salary (RM)", className="card-title"),
            html.H2(f"{avg_min_salary:,.0f}", className="card-text")
        ])
    ], color="success", inverse=True), width=4),
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.H6("Avg. Mean Salary (RM)", className="card-title"),
            html.H2(f"{avg_mean_salary:,.0f}", className="card-text")
        ])
    ], color="warning", inverse=True), width=4),
], className="mt-4 g-4")

# ─── 7) Layout ─────────────────────────────────────────────────────────────────
app.layout = dbc.Container(fluid=True, children=[
    navbar,
    kpi_row,

    dbc.Row([
        # Bar chart with id for callbacks
        dbc.Col(dbc.Card([
            dbc.CardHeader("Jobs by Category"),
            dbc.CardBody(dcc.Graph(id='bar-chart', config={'displayModeBar':False}))
        ], className="h-100 shadow-sm"), md=6),

        # Min salary histogram
        dbc.Col(dbc.Card([
            dbc.CardHeader("Min Salary Distribution"),
            dbc.CardBody(dcc.Graph(id='min-salary-hist', config={'displayModeBar':False}))
        ], className="h-100 shadow-sm"), md=6),
    ], className="mt-4 g-4"),

    dbc.Row([
        # Mean salary histogram
        dbc.Col(dbc.Card([
            dbc.CardHeader("Mean Salary Distribution"),
            dbc.CardBody(dcc.Graph(id='mean-salary-hist', config={'displayModeBar':False}))
        ], className="h-100 shadow-sm"), md=12),
    ], className="mt-4 g-4"),
])

# ─── 8) Initial Figures ─────────────────────────────────────────────────────────
def make_bar_figure():
    fig = px.bar(
        job_counts, x='Category', y='Count',
        template=PX_TEMPLATE
    )
    fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))
    return fig

def make_histograms(filtered_df, cat_label):
    # Min salary
    fig_min = px.histogram(
        filtered_df, x='min_salary', nbins=30,
        template=PX_TEMPLATE
    )
    fig_min.update_layout(
        title_text=f"Min Salary Distribution {cat_label}",
        xaxis_title='Salary (RM)',
        margin=dict(t=40, b=20, l=20, r=20)
    )

    # Mean salary
    fig_mean = px.histogram(
        filtered_df, x='mean_salary', nbins=30,
        template=PX_TEMPLATE
    )
    fig_mean.update_layout(
        title_text=f"Mean Salary Distribution {cat_label}",
        xaxis_title='Salary (RM)',
        margin=dict(t=40, b=20, l=20, r=20)
    )

    return fig_min, fig_mean

# ─── 9) Callbacks to wire charts together ────────────────────────────────────────
@app.callback(
    Output('bar-chart', 'figure'),
    Output('min-salary-hist', 'figure'),
    Output('mean-salary-hist', 'figure'),
    Input('bar-chart', 'clickData')
)
def update_charts(clickData):
    # Always show the bar chart unfiltered
    bar_fig = make_bar_figure()

    if clickData:
        # extract clicked category
        cat = clickData['points'][0]['x']
        dff = df[df['category'] == cat]
        label = f"(Category: {cat})"
    else:
        dff = df
        label = "(All Categories)"

    min_fig, mean_fig = make_histograms(dff, label)
    return bar_fig, min_fig, mean_fig

# ─── 10) Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
