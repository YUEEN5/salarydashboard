import pandas as pd
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px

# ─── Install ──────────────────────────────────────────────────────────────────────
# pip install pandas dash plotly dash-bootstrap-components

# ─── 1) Load Data ────────────────────────────────────────────────────────────────
df = pd.read_csv('dataset/preprocess_dataset2.csv')

# ─── 2) Compute KPIs ─────────────────────────────────────────────────────────────
total_jobs      = len(df)
avg_min_salary  = df['min_salary'].mean()
avg_mean_salary = df['mean_salary'].mean()

# ─── 3) Prepare Aggregations ────────────────────────────────────────────────────
job_counts = (
    df['category']
      .value_counts()
      .reset_index()
)
job_counts.columns = ['Category', 'Count']

# ─── 4) Build Figures ───────────────────────────────────────────────────────────
px_template = 'plotly_white'  # clean white background

fig_jobs = px.bar(
    job_counts, x='Category', y='Count',
    title='Jobs by Category',
    template=px_template
)
fig_jobs.update_layout(margin=dict(t=50, b=10, l=10, r=10))

fig_min_salary = px.histogram(
    df, x='min_salary', nbins=30,
    title='Minimum Salary Distribution',
    template=px_template
)
fig_min_salary.update_layout(
    xaxis_title='Salary (RM)',
    margin=dict(t=50, b=10, l=10, r=10)
)

fig_mean_salary = px.histogram(
    df, x='mean_salary', nbins=30,
    title='Mean Salary Distribution',
    template=px_template
)
fig_mean_salary.update_layout(
    xaxis_title='Salary (RM)',
    margin=dict(t=50, b=10, l=10, r=10)
)

# ─── 5) Initialize App ──────────────────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],  # Flatly theme
    title="JobStreet Dashboard"
)
server = app.server

# ─── 6) Navbar ─────────────────────────────────────────────────────────────────
navbar = dbc.NavbarSimple(
    brand="📈 JobStreet Insights",
    color="primary",
    dark=True,
    fluid=True,
)

# ─── 7) KPI Cards ───────────────────────────────────────────────────────────────
kpi_style = {"textAlign":"center", "padding":"10px"}
kpi = dbc.Row([
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

# ─── 8) Graph Cards Helper ───────────────────────────────────────────────────────
def graph_card(title, fig):
    return dbc.Card([
        dbc.CardHeader(html.H5(title, className="mb-0")),
        dbc.CardBody(dcc.Graph(figure=fig, config={'displayModeBar':False}))
    ], className="h-100 shadow-sm")

# ─── 9) Layout ─────────────────────────────────────────────────────────────────
app.layout = dbc.Container(fluid=True, children=[
    navbar,
    kpi,

    dbc.Row([
        dbc.Col(graph_card("Jobs by Category", fig_jobs), md=6),
        dbc.Col(graph_card("Min Salary Distribution", fig_min_salary), md=6),
    ], className="mt-4 g-4"),

    dbc.Row([
        dbc.Col(graph_card("Mean Salary Distribution", fig_mean_salary), md=12),
    ], className="mt-4 g-4"),
])

# ─── 10) Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
