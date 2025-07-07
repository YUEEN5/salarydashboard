import pandas as pd
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# ─── 1) Load & Prep ─────────────────────────────────────────────────────────────
df = pd.read_csv('dataset/preprocess_dataset2.csv')
PX = 'plotly_white'

# KPI values
total_jobs      = len(df)
avg_min_salary  = df['min_salary'].mean()
avg_mean_salary = df['mean_salary'].mean()

# Dropdown options
categories = sorted(df['category'].unique())
states     = sorted(df['state'].unique())

# ─── 2) Initial Figures ─────────────────────────────────────────────────────────

# 2.1 Pie/Donut: overall postings by Category
counts_init = (
    df
    .groupby('category')
    .size()
    .reset_index(name='count')       # columns ['category','count']
    .rename(columns={'category':'label'})
)
fig_pie_init = px.pie(
    counts_init,
    names='label',
    values='count',
    title="Postings by Category",
    template=PX,
    hole=0.4
)

# 2.2 Bar: Avg Min & Avg Max by Category
avg_min_max = (
    df.groupby('category')[['min_salary','max_salary']]
      .mean()
      .reset_index()
      .rename(columns={'min_salary':'Avg Min','max_salary':'Avg Max'})
)
fig_bar = px.bar(
    avg_min_max,
    x='category',
    y=['Avg Min','Avg Max'],
    barmode='group',
    labels={'value':'Salary (RM)','variable':'Type','category':'Category'},
    title='Avg Min & Max Salary by Category',
    template=PX
)
fig_bar.update_layout(xaxis_tickangle=-45, margin=dict(t=60,b=130,l=40,r=20))

# 2.3 Histograms (full data initial)
min_hist_init  = px.histogram(df, x='min_salary', nbins=30, template=PX)\
                    .update_layout(title_text="Min Salary Distribution (All Categories)", xaxis_title="Salary (RM)")
max_hist_init  = px.histogram(df, x='max_salary', nbins=30, template=PX)\
                    .update_layout(title_text="Max Salary Distribution (All Categories)", xaxis_title="Salary (RM)")
mean_hist_init = px.histogram(df, x='mean_salary', nbins=30, template=PX)\
                    .update_layout(title_text="Mean Salary Distribution (All Categories)", xaxis_title="Salary (RM)")

# 2.4 Salary summary bar (full data initial)
stats_init = pd.DataFrame({
    'Statistic':['Avg Min','Avg Mean','Avg Max'],
    'Salary (RM)':[
        df['min_salary'].mean(),
        df['mean_salary'].mean(),
        df['max_salary'].mean()
    ]
})
fig_summary_init = px.bar(
    stats_init,
    x='Statistic',
    y='Salary (RM)',
    text='Salary (RM)',
    title="Salary Summary (All Data)",
    template=PX
)
fig_summary_init.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
fig_summary_init.update_layout(
    uniformtext_minsize=8,
    yaxis_range=[0, stats_init['Salary (RM)'].max()*1.1],
    margin=dict(t=60,b=20,l=20,r=20)
)

# Empty placeholder for callbacks
empty = go.Figure().update_layout(template=PX, margin=dict(t=40,b=20,l=20,r=20))


# ─── 3) Build App ───────────────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="JobStreet Dashboard")
server = app.server

app.layout = dbc.Container(fluid=True, children=[

    # Navbar + KPIs
    dbc.NavbarSimple(brand="📈 JobStreet Insights", color="dark", dark=True, fluid=True),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Total Postings"), html.H2(f"{total_jobs:,}")]), color="info", inverse=True), width=4),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Avg Min Salary (RM)"), html.H2(f"{avg_min_salary:,.0f}")]), color="success", inverse=True), width=4),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Avg Mean Salary (RM)"), html.H2(f"{avg_mean_salary:,.0f}")]), color="warning", inverse=True), width=4),
    ], className="mt-4 g-4"),

    # ◉ Jobs by Category bar
    dbc.Row(dbc.Col(dbc.Card([
        dbc.CardHeader("Avg Min & Avg Max Salary by Category"),
        dbc.CardBody(dcc.Graph(id='bar-chart', figure=fig_bar, config={'displayModeBar':False}))
    ], className="shadow-sm"), width=12), className="mt-4"),

    # ◉ Salary histograms
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("Min Salary"), dbc.CardBody(dcc.Graph(id='min-salary-hist', figure=min_hist_init, config={'displayModeBar':False}))], className="shadow-sm"), md=4),
        dbc.Col(dbc.Card([dbc.CardHeader("Max Salary"), dbc.CardBody(dcc.Graph(id='max-salary-hist', figure=max_hist_init, config={'displayModeBar':False}))], className="shadow-sm"), md=4),
        dbc.Col(dbc.Card([dbc.CardHeader("Mean Salary"), dbc.CardBody(dcc.Graph(id='mean-salary-hist', figure=mean_hist_init, config={'displayModeBar':False}))], className="shadow-sm"), md=4),
    ], className="mt-4 g-4"),

    html.Hr(),

    # Dropdown filters (no “All”)
    dbc.Row([
        dbc.Col([
            html.Label("Select Category:"),
            dcc.Dropdown(id='ddl-cat', options=[{'label':c,'value':c} for c in categories],
                         placeholder="Select a category", clearable=True)
        ], md=6),
        dbc.Col([
            html.Label("Select State:"),
            dcc.Dropdown(id='ddl-state', options=[{'label':s,'value':s} for s in states],
                         placeholder="Select a state", clearable=True)
        ], md=6),
    ], className="mt-4 g-4"),

# ◉ Pie / Donut chart (first)
    dbc.Row(dbc.Col(dbc.Card([
        dbc.CardHeader("Postings by Category"),
        dbc.CardBody(dcc.Graph(id='pie-chart', figure=fig_pie_init, config={'displayModeBar':False}))
    ], className="shadow-sm"), width=12), className="mt-4"),

    # Salary summary bar
    dbc.Row(dbc.Col(dbc.Card([
        dbc.CardHeader("Salary Summary"),
        dbc.CardBody(dcc.Graph(id='salary-summary', figure=fig_summary_init, config={'displayModeBar':False}))
    ], className="shadow-sm"), width=12), className="mt-4"),
])

# ─── 4) Callbacks ─────────────────────────────────────────────────────────────────

# 4.1 Update histograms on bar‐click
@app.callback(
    Output('min-salary-hist','figure'),
    Output('max-salary-hist','figure'),
    Output('mean-salary-hist','figure'),
    Input('bar-chart','clickData')
)
def update_hists(clickData):
    if clickData:
        cat = clickData['points'][0]['x']
        dff = df[df['category']==cat]
        suffix = f"(Category: {cat})"
    else:
        dff, suffix = df, "(All Categories)"

    fmin = px.histogram(dff, x='min_salary', nbins=30, template=PX)\
             .update_layout(title_text=f"Min Salary Distribution {suffix}", xaxis_title="Salary (RM)")
    fmax = px.histogram(dff, x='max_salary', nbins=30, template=PX)\
             .update_layout(title_text=f"Max Salary Distribution {suffix}", xaxis_title="Salary (RM)")
    fmean = px.histogram(dff, x='mean_salary',nbins=30, template=PX)\
             .update_layout(title_text=f"Mean Salary Distribution {suffix}", xaxis_title="Salary (RM)")
    return fmin, fmax, fmean

# 4.2 Update summary & pie on dropdown‐change
@app.callback(
    Output('salary-summary','figure'),
    Output('pie-chart','figure'),
    Input('ddl-cat','value'),
    Input('ddl-state','value')
)
def update_summary_pie(cat_sel, state_sel):
    dff = df.copy()
    parts = []
    if cat_sel:
        dff = dff[dff['category']==cat_sel]
        parts.append(f"Category: {cat_sel}")
    if state_sel:
        dff = dff[dff['state']==state_sel]
        parts.append(f"State: {state_sel}")
    title = " & ".join(parts) if parts else "All Data"

    # Salary summary
    stats = pd.DataFrame({'Statistic':['Avg Min','Avg Mean','Avg Max'],
                          'Salary (RM)':[dff['min_salary'].mean(), dff['mean_salary'].mean(), dff['max_salary'].mean()]})
    fig_sum = px.bar(stats, x='Statistic', y='Salary (RM)', text='Salary (RM)',
                     title=f"Salary Summary ({title})", template=PX)
    fig_sum.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    fig_sum.update_layout(uniformtext_minsize=8,
                          yaxis_range=[0, stats['Salary (RM)'].max()*1.1])

    # Pie breakdown
    if cat_sel and not state_sel:
        counts = (
            df[df['category']==cat_sel]
            .groupby('state')
            .size()
            .reset_index(name='count')
            .rename(columns={'state':'label'})
        )
        ptitle = f"Postings of {cat_sel} by State"
    elif state_sel and not cat_sel:
        counts = (
            df[df['state']==state_sel]
            .groupby('category')
            .size()
            .reset_index(name='count')
            .rename(columns={'category':'label'})
        )
        ptitle = f"Postings in {state_sel} by Category"
    elif cat_sel and state_sel:
        counts = (
            df[(df['category']==cat_sel)&(df['state']==state_sel)]
            .groupby('type')
            .size()
            .reset_index(name='count')
            .rename(columns={'type':'label'})
        )
        ptitle = f"Types for {cat_sel} in {state_sel}"
    else:
        counts = counts_init  # reuse the initial counts_init DataFrame
        ptitle = "All Postings by Category"

    fig_pie = px.pie(counts, names='label', values='count', title=ptitle, template=PX, hole=0.4)
    return fig_sum, fig_pie

# ─── 5) Run ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
