import pandas as pd
import requests
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px

# ─── 1) Load & Prep ─────────────────────────────────────────────────────────────
df = pd.read_csv('dataset/clean_preprocessed_dataset.csv')

# auto-detect your date column
date_cols = [c for c in df.columns if 'date' in c.lower()]
if not date_cols:
    raise ValueError("No date column found; please include one with 'date' in its name")
date_col = date_cols[0]

PX = 'plotly_white'
month_order = ["Jan","Feb","Mar","Apr","May","Jun"]

# KPI values
total_jobs      = len(df)
avg_min_salary  = df['min_salary'].mean()
avg_mean_salary = df['mean_salary'].mean()
avg_max_salary  = df['max_salary'].mean()

# Dropdown options
categories = sorted(df['category'].unique())
malaysia_states = [
    'Johor','Kedah','Kelantan','Melaka','Negeri Sembilan',
    'Pahang','Penang','Perak','Perlis','Sabah','Sarawak',
    'Selangor','Terengganu','Kuala Lumpur','Labuan','Putrajaya'
]
states   = sorted([s for s in df['state'].unique() if s in malaysia_states])
types    = sorted(df['type'].unique())
roles    = sorted(df['role'].unique())
locations= sorted(df['location'].unique())

# ─── 2) Initial Figures ─────────────────────────────────────────────────────────

# Pie: overall postings by category
counts_init = (
    df.groupby('category').size()
      .reset_index(name='count')
      .rename(columns={'category':'label'})
)
fig_pie_init = px.pie(
    counts_init, names='label', values='count',
    title="All Postings by Category", template=PX, hole=0.4
)

# Bar: Avg Min & Avg Max salary + count per category, custom hover
counts = df.groupby('category').size().reset_index(name='Count')
avg_min_max = (
    df.groupby('category')[['min_salary','max_salary']].mean()
      .reset_index()
      .rename(columns={'min_salary':'Avg Min','max_salary':'Avg Max'})
)
bar_df = (
    avg_min_max
      .melt(id_vars=['category'], value_vars=['Avg Min','Avg Max'],
            var_name='Type', value_name='Salary')
      .merge(counts, on='category')
)
fig_bar = px.bar(
    bar_df,
    x='category', y='Salary', color='Type', barmode='group',
    labels={'Salary':'Salary (RM)'},
    hover_data={
      'Salary': ':.0f',
      'Count': True,
      'category': False,
      'Type': False
    },
    title='Avg Min & Avg Max Salary by Category',
    template=PX
)
fig_bar.update_layout(xaxis_tickangle=-45, margin=dict(t=60,b=130,l=40,r=20))

# ─── 3) Build App ───────────────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="JobStreet Dashboard")
server = app.server

# ─── 4) Layout ───────────────────────────────────────────────────────────────────
app.layout = dbc.Container(fluid=True, children=[

    # Navbar + KPIs
    dbc.NavbarSimple(brand="📈 JobStreet Insights", color="dark", dark=True, fluid=True),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Total Postings"), html.H2(f"{total_jobs:,}")]), color="info", inverse=True), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Avg Min Salary (RM)"), html.H2(f"{avg_min_salary:,.0f}")]), color="success", inverse=True), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Avg Mean Salary (RM)"), html.H2(f"{avg_mean_salary:,.0f}")]), color="warning", inverse=True), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Avg Max Salary (RM)"), html.H2(f"{avg_max_salary:,.0f}")]), color="danger",  inverse=True), width=3),
    ], className="mt-4 g-4"),

    # Avg Min & Avg Max bar chart
    dbc.Row(dbc.Col(dbc.Card([
        dbc.CardHeader("Avg Min & Avg Max Salary by Category"),
        dbc.CardBody(dcc.Graph(id='bar-chart', figure=fig_bar, config={'displayModeBar':False}))
    ], className="shadow-sm"), width=12), className="mt-4 mb-4"),

    # Pie filters & chart
    dbc.Row([
        dbc.Col([html.Label("Select Category:"), dcc.Dropdown(
            id='ddl-cat',
            options=[{'label':c,'value':c} for c in categories],
            placeholder="Select a category", clearable=True
        )], md=6),
        dbc.Col([html.Label("Select State:"), dcc.Dropdown(
            id='ddl-state',
            options=[{'label':s,'value':s} for s in states],
            placeholder="Select a state", clearable=True
        )], md=6),
    ], className="g-4"),
    dbc.Row(dbc.Col(dbc.Card([
        dbc.CardHeader("Postings by Category / State"),
        dbc.CardBody(dcc.Graph(id='pie-chart', figure=fig_pie_init, config={'displayModeBar':False}))
    ], className="shadow-sm"), width=12), className="mt-4 mb-4"),

    # Mean Salary Over Time
    dbc.Card([
        dbc.CardHeader("Mean Salary Over Time"),
        dbc.CardBody([
            dbc.Row([dbc.Col([html.Label("Select Category:"), dcc.Dropdown(
                id="ddl-time-cat",
                options=[{"label":c,"value":c} for c in categories],
                placeholder="Select a category", clearable=True
            )], md=4)], className="mb-3"),
            dcc.Graph(id="time-line", config={"displayModeBar":False})
        ])
    ], className="mt-4 mb-4 shadow-sm"),

    # Salary Prediction
    dbc.Card([
        dbc.CardHeader("Salary Prediction"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([dbc.Label("Category"), dcc.Dropdown(
                    id='pred-category',
                    options=[{'label':c,'value':c} for c in categories],
                    placeholder="Select Category"
                )], md=6),
                dbc.Col([dbc.Label("Role"), dcc.Dropdown(
                    id='pred-role',
                    options=[{'label':r,'value':r} for r in roles],
                    placeholder="Select Role"
                )], md=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([dbc.Label("Location"), dcc.Dropdown(
                    id='pred-location',
                    options=[{'label':loc,'value':loc} for loc in locations],
                    placeholder="Select Location"
                )], md=6),
                dbc.Col([dbc.Label("Job Type"), dcc.Dropdown(
                    id='pred-type',
                    options=[{'label':t,'value':t} for t in types],
                    placeholder="Select Job Type"
                )], md=6),
            ], className="mb-3"),
            dbc.Button("Predict Salary", id='predict-button', color="primary"),
            html.Div(id='prediction-results', className="mt-3")
        ])
    ], className="mt-4 shadow-sm"),

])

# ─── 5) Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output('pie-chart','figure'),
    Input('ddl-cat','value'),
    Input('ddl-state','value')
)
def update_pie(cat_sel, state_sel):
    dff = df.copy()
    if cat_sel:   dff = dff[dff['category']==cat_sel]
    if state_sel:dff = dff[dff['state']==state_sel]

    if cat_sel and not state_sel:
        counts = df[df['category']==cat_sel].groupby('state').size().reset_index(name='count').rename(columns={'state':'label'})
        title = f"Postings of {cat_sel} by State"
    elif state_sel and not cat_sel:
        counts = df[df['state']==state_sel].groupby('category').size().reset_index(name='count').rename(columns={'category':'label'})
        title = f"Postings in {state_sel} by Category"
    elif cat_sel and state_sel:
        counts = df[(df['category']==cat_sel)&(df['state']==state_sel)].groupby('type').size().reset_index(name='count').rename(columns={'type':'label'})
        title = f"Types for {cat_sel} in {state_sel}"
    else:
        counts = counts_init
        title = "All Postings by Category"

    return px.pie(counts, names='label', values='count', title=title, template=PX, hole=0.4)

@app.callback(
    Output("time-line","figure"),
    Input("ddl-time-cat","value")
)
def update_time_line(cat_sel):
    dff = df.copy()
    if cat_sel:
        dff = dff[dff["category"]==cat_sel]
    dff[date_col] = pd.to_datetime(dff[date_col])
    dff["year"]  = dff[date_col].dt.year.astype(str)
    dff["month"] = dff[date_col].dt.month_name().str[:3]

    summary = dff.groupby(["year","month"])["mean_salary"].mean().reset_index()
    pivot  = summary.pivot(index="month", columns="year", values="mean_salary").reindex(month_order)
    plot_df = pivot.reset_index().melt(id_vars="month", var_name="Year", value_name="Mean Salary")

    fig = px.line(
        plot_df, x="month", y="Mean Salary", color="Year",
        markers=True, category_orders={"month":month_order},
        title=f"Mean Salary Over Time ({'All Categories' if not cat_sel else cat_sel})",
        template=PX
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="Mean Salary (RM)",
                      legend=dict(title="Year", orientation="h", y=-0.2))
    return fig

@app.callback(
    Output('prediction-results','children'),
    Input('predict-button','n_clicks'),
    State('pred-category','value'),
    State('pred-role','value'),
    State('pred-location','value'),
    State('pred-type','value'),
    prevent_initial_call=True
)
def predict_salary(n, category, role, location, job_type):
    if not all([category, role, location, job_type]):
        return dbc.Alert("Please fill in all fields to get a prediction.", color="warning")
    payload = {'category':category, 'role':role, 'location':location, 'type':job_type}
    try:
        resp = requests.post('http://localhost:5000/predict', json=payload)
        resp.raise_for_status()
        pred = resp.json()
        return dbc.Card([
            dbc.CardHeader("Predicted Salary"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([html.H6("Min Salary"), html.H4(f"RM {pred['min_salary']:,.0f}", className="text-success")]),
                    dbc.Col([html.H6("Mean Salary"),html.H4(f"RM {pred['mean_salary']:,.0f}", className="text-primary")]),
                    dbc.Col([html.H6("Max Salary"), html.H4(f"RM {pred['max_salary']:,.0f}", className="text-danger")]),
                ], className="mb-3"),
                html.Ul([
                    html.Li(f"Category: {category}"),
                    html.Li(f"Role: {role}"),
                    html.Li(f"Location: {location}"),
                    html.Li(f"Type: {job_type}"),
                ])
            ])
        ], className="shadow-sm")
    except Exception as e:
        return dbc.Alert(f"Error getting prediction: {e}", color="danger")

if __name__ == '__main__':
    app.run(debug=True)
