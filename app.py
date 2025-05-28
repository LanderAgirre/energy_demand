import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import lightgbm as lgb
import holidays
import xgboost as xgb
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
import pickle

EIA_API_KEY = "J3nXv5t8rqB6RkZCrQsLf0VzhGEgjyPJiHXKMIeS"

external_stylesheets = [dbc.themes.SOLAR]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Energy Demand Forecasting"
server = app.server

ENERGY_COLORS = {
    'primary': '#00796B',
    'secondary': '#FFA000',
    'accent': '#0288D1',
    'background': '#F5F5F5',
    'highlight': '#C5E1A5',
    'danger': '#D32F2F',
}

header = dbc.Navbar(
    dbc.Container([
        html.Div([
            html.Img(src='https://img.icons8.com/ios-filled/50/00796B/flash-on.png', style={'height': '40px', 'marginRight': '15px'}),
            html.Span("California Energy Demand Dashboard", style={
                'fontWeight': 'bold', 'fontSize': '2em', 'color': ENERGY_COLORS['primary'],
                'letterSpacing': '1px', 'verticalAlign': 'middle',
            })
        ], style={
            'display': 'flex', 'alignItems': 'center',
            'background': 'linear-gradient(90deg, #C5E1A5 0%, #FFF59D 100%)',
            'padding': '10px 0',
            'borderRadius': '10px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.07)'
        })
    ]),
    color=ENERGY_COLORS['background'],
    dark=False,
    style={'marginBottom': '20px'}
)

def get_energy_data():
    end_date = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=30)
    url = (
        "https://api.eia.gov/v2/electricity/rto/region-data/data/"
        f"?api_key={EIA_API_KEY}"
        f"&frequency=hourly"
        f"&data[0]=value"
        f"&facets[respondent][]=CISO"
        f"&start={start_date.strftime('%Y-%m-%dT%H')}"
        f"&end={end_date.strftime('%Y-%m-%dT%H')}"
    )
    try:
        response = requests.get(url)
        response.raise_for_status()
        data_points = response.json()['response']['data']
        if not data_points:
            return pd.DataFrame()
        df = pd.DataFrame(data_points)
        df['datetime'] = pd.to_datetime(df['period'], errors='coerce')
        df.rename(columns={'value': 'consumption'}, inplace=True)
        df['consumption'] = pd.to_numeric(df['consumption'], errors='coerce')
        import pytz
        tz = pytz.timezone('America/Los_Angeles')
        df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert(tz)
        df['hour'] = df['datetime'].dt.hour
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['is_weekend'] = df['dayofweek'] >= 5
        df = df.groupby('datetime', as_index=False).agg({
            'consumption': 'mean',
            'hour': 'first',
            'dayofweek': 'first',
            'month': 'first',
            'is_weekend': 'first'
        })
        return df[['datetime', 'consumption', 'hour', 'dayofweek', 'month', 'is_weekend']]
    except Exception:
        return pd.DataFrame(columns=['datetime', 'consumption', 'hour', 'dayofweek', 'month', 'is_weekend'])

def add_features(df):
    df = df.sort_values('datetime')
    df['lag1'] = df['consumption'].shift(1)
    df['lag24'] = df['consumption'].shift(24)
    df['lag7'] = df['consumption'].shift(7)
    df['lag168'] = df['consumption'].shift(168)
    df['rolling3'] = df['consumption'].rolling(window=3).mean()
    df['rolling24'] = df['consumption'].rolling(window=24).mean()
    df['rolling168'] = df['consumption'].rolling(window=168).mean()
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['hour_x_month'] = df['hour'] * df['month']
    cal_holidays = holidays.US(state='CA')
    df['is_holiday'] = df['datetime'].dt.date.apply(lambda x: x in cal_holidays)
    q_low = df['consumption'].quantile(0.01)
    q_hi  = df['consumption'].quantile(0.99)
    df = df[(df['consumption'] >= q_low) & (df['consumption'] <= q_hi)]
    df = df.dropna()
    nunique = df.nunique()
    cols_const = nunique[nunique <= 1].index.tolist()
    cols_allnull = df.columns[df.isnull().all()].tolist()
    df = df.drop(columns=cols_const + cols_allnull)
    return df

def train_model(df):
    if df.empty:
        return None, None, None, None, None
    df = add_features(df)
    features = [
        'hour', 'dayofweek', 'month', 'is_weekend', 'lag1', 'lag24', 'lag7', 'lag168',
        'rolling3', 'rolling24', 'rolling168',
        'sin_hour', 'cos_hour', 'sin_month', 'cos_month', 'hour_x_month', 'is_holiday'
    ]
    features = [f for f in features if f in df.columns]
    X = df[features]
    y = df['consumption']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    tscv = TimeSeriesSplit(n_splits=5)
    best_rmse = float('inf')
    best_stack = None
    best_lgb = None
    best_xgb = None
    lgb_params = {'num_leaves': [15, 31], 'learning_rate': [0.01, 0.05], 'n_estimators': [300, 500]}
    xgb_params = {'max_depth': [3, 5], 'learning_rate': [0.01, 0.05], 'n_estimators': [300, 500]}
    for train_idx, test_idx in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        best_lgb_rmse = float('inf')
        for nl in lgb_params['num_leaves']:
            for lr in lgb_params['learning_rate']:
                for ne in lgb_params['n_estimators']:
                    lgb_model = lgb.LGBMRegressor(n_estimators=ne, learning_rate=lr, num_leaves=nl, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
                    try:
                        lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=30, verbose=False)
                    except TypeError:
                        lgb_model.fit(X_train, y_train)
                    lgb_pred = lgb_model.predict(X_test)
                    lgb_rmse = mean_squared_error(y_test, lgb_pred) ** 0.5
                    if lgb_rmse < best_lgb_rmse:
                        best_lgb_rmse = lgb_rmse
                        best_lgb_model = lgb_model
        best_xgb_rmse = float('inf')
        for md in xgb_params['max_depth']:
            for lr in xgb_params['learning_rate']:
                for ne in xgb_params['n_estimators']:
                    xgb_model = xgb.XGBRegressor(n_estimators=ne, learning_rate=lr, max_depth=md, subsample=0.8, colsample_bytree=0.8, random_state=42, objective='reg:squarederror')
                    try:
                        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=30, verbose=False)
                    except TypeError:
                        xgb_model.fit(X_train, y_train)
                    xgb_pred = xgb_model.predict(X_test)
                    xgb_rmse = mean_squared_error(y_test, xgb_pred) ** 0.5
                    if xgb_rmse < best_xgb_rmse:
                        best_xgb_rmse = xgb_rmse
                        best_xgb_model = xgb_model
        lgb_pred = best_lgb_model.predict(X_test)
        xgb_pred = best_xgb_model.predict(X_test)
        stack_X = np.vstack([lgb_pred, xgb_pred]).T
        stacker = RidgeCV(alphas=[0.1, 1.0, 10.0])
        stacker.fit(stack_X, y_test)
        lgb_pred_full = best_lgb_model.predict(X_scaled)
        xgb_pred_full = best_xgb_model.predict(X_scaled)
        stack_X_full = np.vstack([lgb_pred_full, xgb_pred_full]).T
        y_pred = stacker.predict(stack_X_full)
        rmse = mean_squared_error(y, y_pred) ** 0.5
        if rmse < best_rmse:
            best_rmse = rmse
            best_stack = stacker
            best_lgb = best_lgb_model
            best_xgb = best_xgb_model
    lgb_pred_full = best_lgb.predict(X_scaled)
    xgb_pred_full = best_xgb.predict(X_scaled)
    stack_X_full = np.vstack([lgb_pred_full, xgb_pred_full]).T
    y_pred = best_stack.predict(stack_X_full)
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred) ** 0.5
    return (best_stack, best_lgb, best_xgb), scaler, mae, rmse, features

def save_model(model_tuple, scaler, features, filename='model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump({'model_tuple': model_tuple, 'scaler': scaler, 'features': features}, f)

def load_model(filename='model.pkl'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['model_tuple'], data['scaler'], data['features']

df_data = get_energy_data()
model_tuple, scaler, mae, rmse, features = train_model(df_data)
# Guardar el modelo entrenado
save_model(model_tuple, scaler, features)
# Para cargarlo después:
# model_tuple, scaler, features = load_model()

tabs = dcc.Tabs([
    dcc.Tab(label='Last Month Data', children=[
        dbc.Container([
            dbc.Row([
                dbc.Col(html.H3("Energy demand in California last month"))
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col(html.Div([
                    html.H5("App Guide: Tabs Overview"),
                    html.Ul([
                        html.Li([
                            html.B("Forecast: "),
                            "Real-time forecast of hourly energy demand for California (CAISO), with interactive forecast horizon. Shows historical and predicted demand in local time."
                        ]),
                        html.Li([
                            html.B("Last Month Data: "),
                            "Historical hourly demand for the last 30 days. Explore recent consumption patterns and cycles."
                        ]),
                        html.Li([
                            html.B("Analysis: "),
                            "Advanced visualizations (histogram, violin, hourly bar, strip plot) for distribution, seasonality, and variability."
                        ]),
                        html.Li([
                            html.B("Model Details: "),
                            "Feature importance and model summary for transparency."
                        ])
                    ]),
                    html.Br(),
                ]))
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(
                    figure={
                        'data': [go.Scatter(x=df_data['datetime'], y=df_data['consumption'], mode='lines')],
                        'layout': go.Layout(title='Historical Energy Demand (MW)', xaxis={'title': 'Datetime'}, yaxis={'title': 'MW'})
                    }
                ), width=12)
            ]),
            dbc.Row([
                dbc.Col(html.Div([
                    html.P("Hourly electricity demand in California over the last month. Daily and weekly cycles are visible, with peaks in the late afternoon and lows overnight. Use this to spot trends and recurring patterns.")
                ]), width=12)
            ])
        ])
    ]),
    dcc.Tab(label='Analysis', children=[
        dbc.Container([
            dbc.Row([
                dbc.Col(html.H3("Energy Demand Distribution & Variability"), width=12)
            ], className='mb-2'),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='demand-histogram'),
                    html.Div(id='demand-distribution-summary')
                ], md=6),
                dbc.Col([
                    dcc.Graph(id='demand-violin'),
                    html.Div(id='demand-violin-summary')
                ], md=6)
            ], className='mb-2'),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='demand-boxplot'),
                    html.Div(id='demand-outlier-summary')
                ], md=6),
                dbc.Col([
                    dcc.Graph(id='demand-hourly-hist'),
                    html.Div(id='demand-hourly-summary')
                ], md=6)
            ], className='mb-2'),
            dbc.Row([
                dbc.Col([
                    html.H5("Analyst Insights"),
                    html.Div(id='demand-analyst-conclusions', style={'fontSize': '1.1em', 'background': '#f8f9fa', 'padding': '1em', 'borderRadius': '8px'})
                ], width=12)
            ])
        ], fluid=True)
    ]),
    dcc.Tab(label='Model Details', children=[
        dbc.Container([
            dbc.Row([
                dbc.Col(html.H3("Model & Feature Information"))
            ]),
            dbc.Row([
                dbc.Col(html.P(
                    "This app uses a LightGBM/XGBoost stacking ensemble trained on real hourly demand data from the EIA API for CAISO. Features include calendar, lags, rolling means, and holidays. The model is retrained on every update for accuracy."
                ))
            ]),
            dbc.Row([
                dbc.Col(html.H4("Feature Importance (LightGBM)"), width=12)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='feature-importance-graph'))
            ]),
            dbc.Row([
                dbc.Col(html.Div(id='model-details-text'))
            ])
        ], fluid=True)
    ]),
    dcc.Tab(label='Forecast', children=[
        dbc.Container([
            dbc.Row([
                dbc.Col(html.H2("Real-Time Energy Demand Forecast (California Local Time)"), width=12)
            ], className='my-3'),
            dbc.Row([
                dbc.Col([
                    html.Label("Select future hours to forecast:", style={"fontWeight": "bold", "fontSize": "1.1em"}),
                    dcc.Slider(id='hours-slider', min=1, max=24, step=1, value=12,
                               marks={i: str(i) for i in range(1, 25)})
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='live-graph')),
                dcc.Interval(id='interval-component', interval=3600 * 1000, n_intervals=0)
            ]),
            dbc.Row([
                dbc.Col(html.Div(id='model-performance'))
            ])
        ], fluid=True)
    ])
])

app.layout = html.Div([
    header,
    html.Div([
        tabs
    ], style={
        'background': 'linear-gradient(135deg, #F5F5F5 0%, #C5E1A5 40%, #FFF59D 100%)',
        'padding': '30px',
        'borderRadius': '18px',
        'boxShadow': '0 4px 24px rgba(0,0,0,0.10)'
    })
])

@app.callback(
    [Output('live-graph', 'figure'), Output('model-performance', 'children'), Output('live-graph', 'config')],
    [Input('interval-component', 'n_intervals'), Input('hours-slider', 'value')]
)
def update_graph(n, hours):
    df = get_energy_data()
    if df.empty:
        return go.Figure(), "Failed to fetch real-time data.", {}
    model_tuple, scaler, mae, rmse, features = train_model(df)
    if model_tuple is None:
        return go.Figure(), "Model training failed due to insufficient data.", {}
    last_time = df['datetime'].max()
    last_rows = df.sort_values('datetime').tail(24).copy()
    future_times = [last_time + timedelta(hours=i) for i in range(1, hours + 1)]
    future_df = pd.DataFrame({
        'datetime': future_times,
        'hour': [t.hour for t in future_times],
        'dayofweek': [t.dayofweek for t in future_times],
        'month': [t.month for t in future_times],
        'is_weekend': [t.dayofweek >= 5 for t in future_times]
    })
    lags = list(last_rows['consumption'].astype(float).values)
    for i in range(hours):
        lag1 = lags[-1]
        lag24 = lags[-24] if len(lags) >= 24 else lags[0]
        lag7 = lags[-7] if len(lags) >= 7 else lags[0]
        lag168 = lags[-168] if len(lags) >= 168 else lags[0]
        rolling3 = np.mean(lags[-3:]) if len(lags) >= 3 else np.mean(lags)
        rolling24 = np.mean(lags[-24:]) if len(lags) >= 24 else np.mean(lags)
        rolling168 = np.mean(lags[-168:]) if len(lags) >= 168 else np.mean(lags)
        hour = future_df.loc[i, 'hour']
        month = future_df.loc[i, 'month']
        future_df.loc[i, 'lag1'] = lag1
        future_df.loc[i, 'lag24'] = lag24
        future_df.loc[i, 'lag7'] = lag7
        future_df.loc[i, 'lag168'] = lag168
        future_df.loc[i, 'rolling3'] = rolling3
        future_df.loc[i, 'rolling24'] = rolling24
        future_df.loc[i, 'rolling168'] = rolling168
        future_df.loc[i, 'sin_hour'] = np.sin(2 * np.pi * hour / 24)
        future_df.loc[i, 'cos_hour'] = np.cos(2 * np.pi * hour / 24)
        future_df.loc[i, 'sin_month'] = np.sin(2 * np.pi * month / 12)
        future_df.loc[i, 'cos_month'] = np.cos(2 * np.pi * month / 12)
        future_df.loc[i, 'hour_x_month'] = hour * month
        lgb_model = model_tuple[1]
        xgb_model = model_tuple[2]
        stacker = model_tuple[0]
        if 'is_holiday' not in future_df.columns:
            import holidays
            cal_holidays = holidays.US(state='CA')
            future_df['is_holiday'] = future_df['datetime'].dt.date.apply(lambda x: x in cal_holidays)
        pred_features = [f for f in features if f in future_df.columns]
        X_pred = scaler.transform(future_df.loc[[i], pred_features])
        lgb_pred = lgb_model.predict(X_pred)
        xgb_pred = xgb_model.predict(X_pred)
        stack_X = np.vstack([lgb_pred, xgb_pred]).T
        pred = stacker.predict(stack_X)[0]
        lags.append(pred)
        future_df.loc[i, 'prediction'] = pred
    fig = go.Figure()
    y_hist = pd.to_numeric(df['consumption'], errors='coerce')
    y_pred = pd.to_numeric(future_df['prediction'], errors='coerce')
    fig.add_trace(go.Scatter(x=df['datetime'], y=y_hist, mode='lines+markers', name='Historical',
                             line=dict(color=ENERGY_COLORS['primary'], width=2),
                             marker=dict(color=ENERGY_COLORS['highlight'], size=4)))
    fig.add_trace(go.Scatter(x=future_df['datetime'], y=y_pred, mode='lines+markers', name='Forecast',
                             line=dict(color=ENERGY_COLORS['secondary'], width=3, dash='dash'),
                             marker=dict(color=ENERGY_COLORS['accent'], size=6)))
    fig.update_layout(
        title='Observed and Forecasted Energy Demand (MW)',
        xaxis_title='Time',
        yaxis_title='MW',
        plot_bgcolor='#FFF59D',
        paper_bgcolor='#C5E1A5',
        font=dict(family='Segoe UI, Arial', size=15, color=ENERGY_COLORS['primary']),
        legend=dict(bgcolor='#F5F5F5', bordercolor=ENERGY_COLORS['primary'], borderwidth=1)
    )
    perf_text = f"Model MAE: {mae:.2f} MW | RMSE: {rmse:.2f} MW"
    table_config = {'staticPlot': False, 'displayModeBar': True}
    from dash import dash_table
    df_table = df.tail(24).copy()
    df_table['datetime'] = df_table['datetime'].dt.strftime('%Y-%m-%d')
    df_table = df_table.iloc[::-1]
    df_table = df_table.rename(columns={
        'datetime': 'Date',
        'consumption': 'Consumption (MW)',
        'hour': 'Hour',
        'dayofweek': 'Day of Week',
        'month': 'Month',
        'is_weekend': 'Weekend',
        'lag1': 'Lag 1h',
        'lag24': 'Lag 24h',
        'rolling3': 'Rolling 3h Avg',
        'rolling24': 'Rolling 24h Avg'
    })
    display_cols = ['Date', 'Consumption (MW)', 'Hour', 'Day of Week', 'Month', 'Weekend']
    for col in ['Lag 1h', 'Lag 24h', 'Rolling 3h Avg', 'Rolling 24h Avg']:
        if col in df_table.columns:
            display_cols.append(col)
    df_table = df_table[display_cols]
    for col in ['Consumption (MW)', 'Lag 1h', 'Lag 24h', 'Rolling 3h Avg', 'Rolling 24h Avg']:
        if col in df_table.columns:
            df_table[col] = df_table[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
    table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in df_table.columns],
        data=df_table.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center', 'fontFamily': 'Segoe UI, Arial', 'fontSize': '1em', 'color': '#222'},
        style_header={'backgroundColor': ENERGY_COLORS['primary'], 'color': '#FFF', 'fontWeight': 'bold'},
        style_data={'backgroundColor': ENERGY_COLORS['background']},
    )
    import io
    import base64
    def generate_download_link(df):
        csv_string = df.to_csv(index=False, encoding='utf-8')
        b64 = base64.b64encode(csv_string.encode()).decode()
        return html.A(
            'Download full data as CSV',
            id='download-link',
            download='energy_demand_full.csv',
            href=f'data:text/csv;base64,{b64}',
            target='_blank',
            style={'margin': '10px', 'fontWeight': 'bold', 'color': ENERGY_COLORS['primary'], 'textDecoration': 'underline'}
        )
    store = dcc.Store(id='df-store', data=df.to_dict('records'))

    return fig, [perf_text, html.Br(), html.B("Last 24 hours of historical data:"), table, html.Br(), generate_download_link(df), store], table_config

@app.callback(
    [Output('feature-importance-graph', 'figure'), Output('model-details-text', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_feature_importance(n):
    df = get_energy_data()
    model_tuple, scaler, mae, rmse, features = train_model(df)
    if model_tuple is None:
        return go.Figure(), "Model not trained."
    lgb_model = model_tuple[1]
    lgb_imp = lgb_model.feature_importances_
    imp_df = pd.DataFrame({'feature': features, 'importance': lgb_imp})
    imp_df = imp_df.sort_values('importance', ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=imp_df['feature'], y=imp_df['importance'], name='LightGBM',
                         marker_color=[ENERGY_COLORS['primary'], ENERGY_COLORS['secondary'], ENERGY_COLORS['accent'], ENERGY_COLORS['highlight']] * 5))
    fig.update_layout(barmode='group', xaxis_title='Feature', yaxis_title='Importance',
                      plot_bgcolor='#FFF59D', paper_bgcolor='#C5E1A5',
                      font=dict(family='Segoe UI, Arial', size=15, color=ENERGY_COLORS['primary']))
    text = html.Div([
        html.P(f"MAE: {mae:.2f} MW | RMSE: {rmse:.2f} MW"),
        html.P(
            "The bar chart shows the relative importance of each predictor variable according to LightGBM, sorted from most to least relevant. "
            "Variables with higher importance contribute more to the hourly demand prediction. Interpretation is based solely on LightGBM."
        )
    ])
    return fig, text

@app.callback(
    [Output('demand-histogram', 'figure'),
     Output('demand-distribution-summary', 'children'),
     Output('demand-boxplot', 'figure'),
     Output('demand-outlier-summary', 'children'),
     Output('demand-violin', 'figure'),
     Output('demand-violin-summary', 'children'),
     Output('demand-hourly-hist', 'figure'),
     Output('demand-hourly-summary', 'children'),
     Output('demand-analyst-conclusions', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_demand_dashboard(n):
    df = get_energy_data()
    if df.empty:
        empty_fig = go.Figure()
        return [empty_fig, "No data available."] * 8 + ["No data available."]
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=df['consumption'], nbinsx=40, name='Demand Distribution',
                                    marker_color=ENERGY_COLORS['primary'], opacity=0.75))
    hist_fig.update_layout(title='Distribution of Energy Demand (MW)', xaxis_title='MW', yaxis_title='Frequency',
                          plot_bgcolor='#FFF59D', paper_bgcolor='#C5E1A5',
                          font=dict(family='Segoe UI, Arial', size=15, color=ENERGY_COLORS['primary']))
    hist_summary = html.Div([
        html.B("Histogram: Distribution of energy demand (MW)", style={"color": "#333"}),
        html.Br(),
        html.Span("Visualiza la frecuencia de la demanda horaria en el periodo seleccionado. Permite identificar los rangos más habituales, asimetrías y posibles valores atípicos de forma rápida.", style={"color": "#333"})
    ])
    desc = df['consumption'].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
    violin_fig = go.Figure()
    violin_fig.add_trace(go.Violin(x=df['month'], y=df['consumption'], box_visible=True, meanline_visible=True,
                                   line_color=ENERGY_COLORS['accent'], fillcolor=ENERGY_COLORS['highlight'], opacity=0.7, name='By Month'))
    violin_fig.update_layout(title='Distribution of Demand for the last two months', xaxis_title='Month', yaxis_title='MW',
                            plot_bgcolor='#FFF59D', paper_bgcolor='#C5E1A5',
                            font=dict(family='Segoe UI, Arial', size=15, color=ENERGY_COLORS['primary']))
    violin_summary = html.Div([
        html.B("Violin plot: Variability by month", style={"color": "#333"}),
        html.Br(),
        html.Span("Shows how demand distribution changes seasonally.", style={"color": "#333"})
    ])
    hourly_data = df.groupby('hour')['consumption'].mean().reset_index()
    hourly_fig = go.Figure()
    hourly_fig.add_trace(go.Bar(x=hourly_data['hour'], y=hourly_data['consumption'], marker_color=ENERGY_COLORS['secondary']))
    hourly_fig.update_layout(title='Average Energy Demand by Hour of Day (MW)', xaxis_title='Hour of Day', yaxis_title='MW',
                            plot_bgcolor='#FFF59D', paper_bgcolor='#C5E1A5',
                            font=dict(family='Segoe UI, Arial', size=15, color=ENERGY_COLORS['primary']))
    hourly_summary = html.Div([
        html.B("Bar chart: Mean demand by hour of day", style={"color": "#333"}),
        html.Br(),
        html.Span("Shows the typical daily demand pattern and peak hours.", style={"color": "#333"})
    ])
    scatter_fig = go.Figure()
    scatter_fig.add_trace(go.Box(
        x=df['hour'],
        y=df['consumption'],
        boxpoints='all',
        jitter=0.3,
        marker_color=ENERGY_COLORS['danger'],
        fillcolor=ENERGY_COLORS['highlight'],
        name='Hourly Demand Spread',
        orientation='v',
        showlegend=False
    ))
    scatter_fig.update_layout(title='Hourly Demand Variability (Strip Plot)', xaxis_title='Hour of Day', yaxis_title='MW',
                             plot_bgcolor='#FFF59D', paper_bgcolor='#C5E1A5',
                             font=dict(family='Segoe UI, Arial', size=15, color=ENERGY_COLORS['primary']))
    scatter_summary = html.Div([
        html.B("Strip plot: Demand variability by hour of day", style={"color": "#333"}),
        html.Br(),
        html.Span("Visualizes the spread and outliers of demand for each hour, complementing the mean bar chart.", style={"color": "#333"})
    ])
    insights = [
        "The hourly bar chart reveals a clear daily cycle, with pronounced peaks during the afternoon and lower demand overnight.",
        "The strip plot highlights the variability of demand at each hour, showing that some hours are consistently more volatile than others.",
        "These insights can support operational planning, resource allocation, and the design of demand response strategies."
    ]
    analyst_conclusions = html.Ul([html.Li(i) for i in insights])
    return hist_fig, hist_summary, scatter_fig, scatter_summary, violin_fig, violin_summary, hourly_fig, hourly_summary, analyst_conclusions

if __name__ == '__main__':
    app.run(debug=True)