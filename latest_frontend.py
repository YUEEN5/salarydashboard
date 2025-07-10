import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import requests  # For making API calls to your backend

# ─── 1) Load Data ────────────────────────────────────────────────────────────────
df = pd.read_csv('dataset/preprocess_dataset3.csv')

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

# Get unique values for dropdown options
job_titles = sorted(df['job_title'].unique())
categories = sorted(df['category'].unique())
roles = sorted(df['role'].unique())
locations = sorted(df['location'].unique())
types = sorted(df['type'].unique())

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

# ─── 7) Salary Prediction Form ─────────────────────────────────────────────────
prediction_form = dbc.Card([
    dbc.CardHeader("Salary Prediction"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dbc.Label("Job Title"),
                dcc.Dropdown(
                    id='job-title-dropdown',
                    options=[{'label': title, 'value': title} for title in job_titles],
                    placeholder="Select Job Title"
                ),
            ], md=6),
            dbc.Col([
                dbc.Label("Category"),
                dcc.Dropdown(
                    id='category-dropdown',
                    options=[{'label': cat, 'value': cat} for cat in categories],
                    placeholder="Select Category"
                ),
            ], md=6),
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                dbc.Label("Role"),
                dcc.Dropdown(
                    id='role-dropdown',
                    options=[{'label': role, 'value': role} for role in roles],
                    placeholder="Select Role"
                ),
            ], md=6),
            dbc.Col([
                dbc.Label("Location"),
                dcc.Dropdown(
                    id='location-dropdown',
                    options=[{'label': loc, 'value': loc} for loc in locations],
                    placeholder="Select Location"
                ),
            ], md=6),
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                dbc.Label("Job Type"),
                dcc.Dropdown(
                    id='type-dropdown',
                    options=[{'label': typ, 'value': typ} for typ in types],
                    placeholder="Select Job Type"
                ),
            ], md=6),
            dbc.Col([
                dbc.Button("Predict Salary", id='predict-button', color="primary", className="mt-4"),
            ], md=6),
        ]),
        
        # Prediction results will be displayed here
        html.Div(id='prediction-results', className="mt-4")
    ])
], className="mt-4 shadow-sm")

# ─── 8) Layout ─────────────────────────────────────────────────────────────────
app.layout = dbc.Container(fluid=True, children=[
    navbar,
    kpi_row,
    
    # Add the prediction form at the top
    prediction_form,

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

# ─── 9) Initial Figures ─────────────────────────────────────────────────────────
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

# ─── 10) Callbacks to wire charts together ────────────────────────────────────────
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

# ─── 11) Salary Prediction Callback ──────────────────────────────────────────────
@app.callback(
    Output('prediction-results', 'children'),
    Input('predict-button', 'n_clicks'),
    State('job-title-dropdown', 'value'),
    State('category-dropdown', 'value'),
    State('role-dropdown', 'value'),
    State('location-dropdown', 'value'),
    State('type-dropdown', 'value'),
    prevent_initial_call=True
)
def predict_salary(n_clicks, job_title, category, role, location, job_type):
    if not all([job_title, category, role, location, job_type]):
        return dbc.Alert("Please fill in all fields to get a prediction.", color="warning")
    
    # Prepare the data to send to your backend API
    input_data = {
        'job_title': job_title,
        'category': category,
        'role': role,
        'location': location,
        'type': job_type
    }
    
    try:
        # Replace with your actual backend API endpoint
        # Replace this line in your Dash frontend:
        response = requests.post('http://localhost:5000/predict', json=input_data)
        response.raise_for_status()
        prediction = response.json()
        
        # Assuming your backend returns min, mean, max salary predictions
        return dbc.Card([
            dbc.CardHeader("Predicted Salary"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Minimum Salary", className="card-title"),
                        html.H4(f"RM {prediction.get('min_salary', 0):,.0f}", className="text-success")
                    ]),
                    dbc.Col([
                        html.H6("Average Salary", className="card-title"),
                        html.H4(f"RM {prediction.get('mean_salary', 0):,.0f}", className="text-primary")
                    ]),
                    dbc.Col([
                        html.H6("Maximum Salary", className="card-title"),
                        html.H4(f"RM {prediction.get('max_salary', 0):,.0f}", className="text-danger")
                    ]),
                ]),
                html.Hr(),
                html.P("Based on your selection:", className="text-muted"),
                html.Ul([
                    html.Li(f"Job Title: {job_title}"),
                    html.Li(f"Category: {category}"),
                    html.Li(f"Role: {role}"),
                    html.Li(f"Location: {location}"),
                    html.Li(f"Type: {job_type}"),
                ])
            ])
        ])
    except Exception as e:
        return dbc.Alert(f"Error getting prediction: {str(e)}", color="danger")

# ─── 12) Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
