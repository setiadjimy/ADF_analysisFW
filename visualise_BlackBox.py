import base64
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import cantools
import dash
from dash import dcc, html, Input, Output, State, DiskcacheManager
from dash.exceptions import PreventUpdate
import diskcache

# --- BACKGROUND CALLBACK MANAGER ---
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

# --- CONSTANTS ---
MAX_GRAPH_POINTS = 2000
EXPECTED_FIRMWARE_VERSION = "5555"

# --- LOG FORMAT ALIGNMENT ---
# Based on: 2026-03-27 09:18:05.585, 2026-03-27 09:18:05.585, -0.000683, -0.000237, 019E, 0000000000000000, 8
LOG_COLUMN_NAMES = ['sys_time', 'timestamp', 'gps_lat', 'gps_lon', 'can_id', 'can_data', 'can_dlc']

# --- DASH APP INITIALIZATION ---
app = dash.Dash(__name__, background_callback_manager=background_callback_manager, suppress_callback_exceptions=True)
app.title = "Vehicle Telemetry & IDS Dashboard"

# --- STYLING & LAYOUT ---
app.layout = html.Div(style={'backgroundColor': '#111111', 'color': '#FFFFFF', 'fontFamily': 'sans-serif', 'minHeight': '100vh'}, children=[
    # Header
    html.Div(
        className="header",
        style={'padding': '20px', 'textAlign': 'center', 'backgroundColor': '#222222'},
        children=[
            html.H1("Vehicle Telemetry & IDS Dashboard", style={'margin': '0'}),
            html.H4("Academic PoC: Cyber-Physical Analysis & UDS Validation")
        ]
    ),

    # File Upload Section
    html.Div(
        className="upload-section",
        style={'padding': '20px', 'margin': '20px auto', 'border': '1px dashed #555', 'borderRadius': '5px', 'maxWidth': '1000px'},
        children=[
            html.Div(style={'display': 'flex', 'justifyContent': 'space-around', 'alignItems': 'center'}, children=[
                dcc.Upload(id='upload-log-data', children=html.Div(['Drag and Drop or ', html.A('Select Log File')]), style={'width': '45%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px', 'cursor': 'pointer'}),
                dcc.Upload(id='upload-dbc-data', children=html.Div(['Drag and Drop or ', html.A('Select DBC File')]), style={'width': '45%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px', 'cursor': 'pointer'}),
            ]),
            html.Div(id='output-log-filename', style={'marginTop': '10px', 'textAlign': 'center'}),
            html.Div(id='output-dbc-filename', style={'marginTop': '5px', 'textAlign': 'center'}),
            html.Div(style={'textAlign': 'center', 'marginTop': '20px'}, children=[html.Button('Start Analysis', id='start-button', n_clicks=0, disabled=True, style={'padding': '10px 20px', 'fontSize': '16px', 'cursor': 'pointer'})]),
            html.Div(id='progress-bar-container', style={'marginTop': '20px', 'padding': '0 10%', 'visibility': 'hidden'}, children=[
                html.Div(id='progress-bar-outer', style={'width': '100%', 'backgroundColor': '#444', 'borderRadius': '5px', 'height': '20px', 'overflow': 'hidden'}, children=[
                    html.Div(id='progress-bar-inner', style={'width': '0%', 'backgroundColor': '#76b852', 'height': '20px', 'textAlign': 'center', 'color': 'white', 'transition': 'width 0.1s ease-in-out'}, children='0%')
                ])
            ]),
            html.Div(id='output-status-message', style={'marginTop': '10px', 'textAlign': 'center'})
        ]
    ),

    # Main Dashboard (Initially hidden)
    html.Div(id='dashboard-content', style={'display': 'none'}, children=[
        html.Div(className="controls-and-kpi-container", style={'display': 'flex', 'flexWrap': 'wrap', 'padding': '20px'}, children=[
            html.Div(id='signal-controls', style={'width': '100%', 'md': {'width': '25%'}, 'padding': '10px', 'backgroundColor': '#222', 'borderRadius': '5px'}, children=[
                html.H4("Select Signals to Plot"),
                dcc.Checklist(id='signal-checklist', options=[], value=[], labelStyle={'display': 'block', 'margin': '5px'}),
                html.Button('Plot Selected Signals', id='plot-button', n_clicks=0, style={'marginTop': '15px', 'width': '100%'})
            ]),
            html.Div(id='kpi-row', style={'width': '100%', 'md': {'width': '75%'}, 'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap', 'padding': '10px'}),
        ]),
        
        # Analysis Container
        html.Div(className="analysis-container", style={'display': 'flex', 'flexWrap': 'wrap', 'padding': '0 20px'}, children=[
            html.Div(style={'width': '50%', 'paddingRight': '10px'}, children=[
                html.H3("Vehicle Cyber-Physical Behavior", style={'textAlign': 'center'}),
                html.Div(id='behavior-analysis-output', style={'padding': '15px', 'backgroundColor': '#222', 'borderRadius': '5px', 'height': '400px', 'overflowY': 'auto'})
            ]),
            html.Div(style={'width': '50%', 'paddingLeft': '10px'}, children=[
                html.H3("UDS Protocol Validation", style={'textAlign': 'center'}),
                html.Div(id='uds-auth-output', style={'padding': '15px', 'backgroundColor': '#222', 'borderRadius': '5px', 'height': '400px', 'overflowY': 'auto'})
            ])
        ]),

        # Graph & Map Container
        html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'padding': '20px'}, children=[
            html.Div(id='graphs-container', style={'width': '100%', 'lg': {'width': '60%'}}),
            dcc.Graph(id='gps-map', style={'width': '100%', 'lg': {'width': '40%'}, 'height': '50vh'}),
        ]),
    ]),

    # Data Stores (Removed stored-log-contents to prevent memory crash)
    dcc.Store(id='processed-data-store'),
    dcc.Store(id='active-signals-store'),
    dcc.Store(id='hover-time-store'),
    dcc.Store(id='dbc-store'), 
])

# --- DATA PROCESSING FUNCTIONS ---

def parse_log_and_dbc(set_progress, log_contents, dbc_contents):
    if not log_contents or not dbc_contents:
        return None, None, None

    try:
        log_content_type, log_content_string = log_contents.split(',')
        dbc_content_type, dbc_content_string = dbc_contents.split(',')

        decoded_log = base64.b64decode(log_content_string)
        decoded_dbc = base64.b64decode(dbc_content_string)

        db = cantools.database.load_string(decoded_dbc.decode('utf-8'))
        
        # Parse CSV matching the 7-column format provided
        df = pd.read_csv(io.StringIO(decoded_log.decode('utf-8')), header=None, names=LOG_COLUMN_NAMES)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp']) 
        df.set_index('timestamp', inplace=True)

        decoded_signals = []
        total_rows = len(df)

        for i, (index, row) in enumerate(df.iterrows()):
            try:
                # Handle Hex strings from the CSV (e.g., '019E')
                can_id_int = int(str(row['can_id']).strip(), 16)
                can_data_str = str(row['can_data']).strip()
                if len(can_data_str) % 2 != 0:
                    can_data_str = '0' + can_data_str
                can_data_bytes = bytes.fromhex(can_data_str)
                decoded_message = db.decode_message(can_id_int, can_data_bytes)
                decoded_signals.append(decoded_message)
            except Exception:
                decoded_signals.append({})

            # Update progress
            if i % 100 == 0 or i == total_rows - 1:
                progress = int((i + 1) / total_rows * 100)
                progress_style = {
                    'width': f'{progress}%', 
                    'backgroundColor': '#76b852', 
                    'height': '20px', 
                    'textAlign': 'center', 
                    'color': 'white', 
                    'transition': 'width 0.1s ease-in-out'
                }
                set_progress((progress_style, f'{progress}%'))

        decoded_df = pd.DataFrame(decoded_signals, index=df.index)
        
        # Merge decoded signals with raw data (retaining can_id and can_data for UDS)
        df = pd.concat([df, decoded_df], axis=1)

        # Forward fill missing values for plotting
        for col in decoded_df.columns:
            if col in df.columns:
                df[col] = df[col].ffill()

        # Find active signals
        numeric_cols = [col for col in decoded_df.columns if pd.api.types.is_numeric_dtype(df[col])]
        active_signals = []
        for col in numeric_cols:
            if col in df.columns and not df[col].isna().all():
                col_min, col_max = df[col].min(), df[col].max()
                if pd.notna(col_min) and pd.notna(col_max) and col_min != col_max:
                    active_signals.append(col)

        return df, active_signals, dbc_contents

    except Exception as e:
        print(f"Error processing files: {e}")
        return None, None, None

def downsample_data(df, signals):
    if df.empty or not signals:
        return pd.DataFrame()
    available_signals = [sig for sig in signals if sig in df.columns]
    if not available_signals:
        return pd.DataFrame()
    subset_df = df[available_signals].copy().dropna(how='all')
    if subset_df.empty:
        return pd.DataFrame()

    total_rows = len(subset_df)
    if total_rows <= MAX_GRAPH_POINTS:
        return subset_df

    duration_seconds = (subset_df.index.max() - subset_df.index.min()).total_seconds()
    if duration_seconds > 0:
        resample_rule = f"{max(1, int(duration_seconds / MAX_GRAPH_POINTS))}s"
        resampled_df = subset_df.resample(resample_rule).mean()
        return resampled_df.dropna(how='all')
    else:
        sample_rate = max(1, total_rows // MAX_GRAPH_POINTS)
        return subset_df.iloc[::sample_rate]

# --- ANALYSIS FUNCTIONS ---

def analyze_vehicle_behavior_with_dbc(df, dbc_contents):
    """
    Analyze vehicle behavior: Crash detection (rapid deceleration) & Sensor anomalies
    """
    try:
        dbc_content_type, dbc_content_string = dbc_contents.split(',')
        decoded_dbc = base64.b64decode(dbc_content_string)
        db = cantools.database.load_string(decoded_dbc.decode('utf-8'))
        
        brake_signal, speed_signal, engine_torque_signal = None, None, None
        
        for message in db.messages:
            for signal in message.signals:
                if signal.name == 'BrakePressed': brake_signal = signal.name
                elif signal.name == 'VehicleSpeed': speed_signal = signal.name
                elif signal.name == 'EngineTorque': engine_torque_signal = signal.name
        
        analysis_summary = "Kinematic & Behavior Analysis:\n==========================\n\n"
        
        # Document signals found
        analysis_summary += f"{'✓' if brake_signal in df.columns else '⚠'} Brake signal: {brake_signal}\n"
        analysis_summary += f"{'✓' if speed_signal in df.columns else '⚠'} Speed signal: {speed_signal}\n"
        
        # 1. Crash Detection (Rapid Deceleration)
        if speed_signal in df.columns:
            analysis_summary += f"\nCrash Detection (Kinematics):\n=====================================\n"
            speed_data = df[speed_signal].dropna().groupby(level=0).mean()
            
            if not speed_data.empty:
                speed_diff = speed_data.diff()
                time_diff = speed_data.index.to_series().diff().dt.total_seconds()
                speed_rate = speed_diff / time_diff.replace(0, 1e-6) # km/h/s
                
                rapid_decel_threshold = -151.0
                rapid_decel_events = speed_rate[speed_rate < rapid_decel_threshold]
                
                if not rapid_decel_events.empty:
                    analysis_summary += f"⚠ CRASH DETECTED: {len(rapid_decel_events)} events exceeding {rapid_decel_threshold} km/h/s\n\n"
                    worst_events = rapid_decel_events.nsmallest(3)
                    
                    for i, (timestamp, decel_rate) in enumerate(worst_events.items(), 1):
                        decel_val = float(decel_rate.iloc[0]) if hasattr(decel_rate, 'iloc') else float(decel_rate)
                        analysis_summary += f"  {i}. Time: {timestamp.strftime('%H:%M:%S.%f')[:-3]} | Rate: {decel_val:.2f} km/h/s\n"
                else:
                    analysis_summary += "✓ No critical impact/deceleration events detected.\n"
        
        # 2. Cyber-Physical Anomaly (Brake vs Speed)
        if brake_signal in df.columns and speed_signal in df.columns:
            analysis_summary += f"\nCyber-Physical Integrity (Brake/Speed):\n------------------------\n"
            brake_pressed = df[df[brake_signal] > 0]
            
            if not brake_pressed.empty:
                speed_during_braking = df.loc[brake_pressed.index, speed_signal].dropna()
                if not speed_during_braking.empty:
                    speed_changes = speed_during_braking.diff().dropna()
                    time_changes = speed_during_braking.index.to_series().diff().dropna()
                    speed_rate_change = speed_changes / time_changes.dt.total_seconds().replace(0, 1e-6)
                    
                    decreasing = (speed_rate_change < 0).sum()
                    increasing = (speed_rate_change > 0).sum()
                    
                    analysis_summary += f"• Brake applied {len(brake_pressed)} times.\n"
                    if increasing > decreasing:
                        analysis_summary += "⚠ ANOMALY DETECTED: Speed increased more often than it decreased during braking!\n"
                        analysis_summary += "  Potential causes: Sensor spoofing (CAN Injection) or mechanical failure.\n"
                    else:
                        analysis_summary += "✓ Cyber-physical behavior normal (Speed reliably decreases during braking).\n"
            else:
                analysis_summary += "• Brake was not applied during log duration.\n"

        return analysis_summary
    except Exception as e:
        return f"Error in vehicle behavior analysis: {str(e)}"

def check_uds_authentication_sequence(df):
    try:
        valid_uds_ids = ['0700', '700', '07df', '7df', '05c6', '5c6', '1792', '1478']
        uds_messages = df[df['can_id'].astype(str).str.lower().str.strip().isin(valid_uds_ids)].copy()

        if uds_messages.empty:
            return "UDS Authentication Table:\n========================\n\nNo diagnostic messages found.\n"

        sessions = []

        for index, row in uds_messages.iterrows():
            try:
                can_id = str(row['can_id']).strip()
                can_data_bytes = bytes.fromhex(str(row['can_data']).strip().zfill(16))

                if len(can_data_bytes) < 2:
                    continue

                service_id = can_data_bytes[0]
                subfunction = can_data_bytes[1]
                timestamp = index.strftime('%H:%M:%S')

                # --- START SESSION ---
                if service_id == 0x27 and subfunction == 0x01:
                    sessions.append({
                        "Timestamp": timestamp,
                        "Frame ID": can_id,
                        "authenticated": False,
                        "firmware": None,
                        "saw_seed_response": False,
                        "saw_key": False
                    })

                if not sessions:
                    continue

                current = sessions[-1]

                # Track steps
                if service_id == 0x67 and subfunction == 0x01:
                    current["saw_seed_response"] = True

                elif service_id == 0x27 and subfunction == 0x02:
                    current["saw_key"] = True

                elif service_id == 0x67 and subfunction == 0x02:
                    current["authenticated"] = True

                elif service_id == 0x22:
                    data = can_data_bytes[3:]
                    try:
                        firmware = data.decode('utf-8', errors='ignore').strip('\x00')
                    except:
                        firmware = data.hex()
                    current["firmware"] = firmware

            except Exception:
                continue

        # --- CLASSIFY SESSIONS ---
        results = []

        for s in sessions:
            # CASE 4: ECU UNRESPONSIVE (no seed response at all)
            if not s["saw_seed_response"]:
                response = "ECU_UNRESPONSIVE"

            # CASE 3: AUTH FAILED (handshake attempted but no success)
            elif not s["authenticated"]:
                response = "AUTHENTICATION_UNSUCCESSFUL"

            # CASE 2: Firmware mismatch
            elif s["firmware"] and str(s["firmware"]) != str(EXPECTED_FIRMWARE_VERSION):
                response = "FIRMWARE_MISMATCH"

            # CASE 1: Success
            elif s["authenticated"]:
                response = "SUCCESSFUL"

            else:
                response = "UNKNOWN"

            results.append({
                "Timestamp": s["Timestamp"],
                "Frame ID": s["Frame ID"],
                "Response": response,
                "Firmware": s["firmware"] if s["firmware"] else "N/A"
            })

        # --- BUILD TABLE ---
        output = "UDS Authentication Table:\n========================\n\n"
        output += f"{'Timestamp':<12} | {'Frame ID':<8} | {'Response':<28} | {'Firmware'}\n"
        output += "-" * 70 + "\n"

        for row in results:
            output += f"{row['Timestamp']:<12} | {row['Frame ID']:<8} | {row['Response']:<28} | {row['Firmware']}\n"

        return output

    except Exception as e:
        return f"Error in UDS analysis: {str(e)}"

def create_map_and_kpis(df, active_signals, hover_time):
    if df.empty: return go.Figure(), []
    try:
        if hover_time:
            hover_timestamp = pd.to_datetime(hover_time)
            time_diffs = abs(df.index - hover_timestamp)
            closest_index = df.index[time_diffs.argmin()]
            current_point = df.loc[closest_index]
        else:
            current_point = df.iloc[-1]

        kpi_children = []
        if 'gps_lat' in df.columns and 'gps_lon' in df.columns and 'gps_lat' in current_point.index:
            if pd.notna(current_point['gps_lat']) and pd.notna(current_point['gps_lon']):
                kpi_children.append(
                    html.Div([
                        html.H3("Location"),
                        html.P(f"Lat: {current_point['gps_lat']:.6f}"),
                        html.P(f"Lon: {current_point['gps_lon']:.6f}")
                    ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#222', 'borderRadius': '5px', 'margin': '5px'})
                )

        if active_signals:
            for col in active_signals:
                if col in current_point.index and pd.notna(current_point[col]):
                    kpi_children.append(
                        html.Div([
                            html.H3(col),
                            html.P(f"{current_point[col]:.2f}")
                        ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#222', 'borderRadius': '5px', 'margin': '5px'})
                    )

        fig_map = go.Figure()
        if 'gps_lat' in df.columns and 'gps_lon' in df.columns:
            gps_df = df[['gps_lat', 'gps_lon']].dropna()
            if not gps_df.empty:
                fig_map.add_trace(go.Scattermapbox(
                    lat=gps_df['gps_lat'], lon=gps_df['gps_lon'], mode='lines',
                    line=dict(width=2, color='#76b852'), name='GPS Track'
                ))
                if 'gps_lat' in current_point.index and pd.notna(current_point['gps_lat']):
                    fig_map.add_trace(go.Scattermapbox(
                        lat=[current_point['gps_lat']], lon=[current_point['gps_lon']],
                        mode='markers', marker=go.scattermapbox.Marker(size=14, color='red'), name='Position'
                    ))
                fig_map.update_layout(mapbox_style="carto-darkmatter", mapbox=dict(zoom=15, center={"lat": gps_df['gps_lat'].mean(), "lon": gps_df['gps_lon'].mean()}))

        fig_map.update_layout(title="Vehicle GPS Track", template='plotly_dark', margin={"r": 0, "t": 40, "l": 0, "b": 0}, showlegend=False)
        return fig_map, kpi_children
    except Exception as e:
        print(f"Map error: {e}")
        return go.Figure(), []

# --- CALLBACKS ---

@app.callback(
    Output('output-log-filename', 'children'),
    Output('output-dbc-filename', 'children'),
    Output('start-button', 'disabled'),
    Input('upload-log-data', 'filename'),
    Input('upload-dbc-data', 'filename')
)
def update_filenames(log_filename, dbc_filename):
    return (
        f"Log File: {log_filename}" if log_filename else "Not selected",
        f"DBC File: {dbc_filename}" if dbc_filename else "Not selected",
        not (log_filename and dbc_filename)
    )

@app.callback(
    Output('processed-data-store', 'data'),
    Output('active-signals-store', 'data'),
    Output('dbc-store', 'data'),
    Output('output-status-message', 'children'),
    Input('start-button', 'n_clicks'),
    State('upload-log-data', 'contents'),
    State('upload-dbc-data', 'contents'),
    background=True,
    progress=[Output('progress-bar-inner', 'style'), Output('progress-bar-inner', 'children')],
    running=[
        (Output("start-button", "disabled"), True, False),
        (Output("progress-bar-container", "style"), {'visibility': 'visible', 'marginTop': '20px', 'padding': '0 10%'}, {'visibility': 'hidden', 'marginTop': '20px', 'padding': '0 10%'}),
    ],
    prevent_initial_call=True
)
def run_analysis(set_progress, n_clicks, log_contents, dbc_contents):
    if n_clicks > 0:
        df, active_signals, dbc_data = parse_log_and_dbc(set_progress, log_contents, dbc_contents)
        if df is not None:
            return df.to_json(date_format='iso', orient='split'), active_signals, dbc_data, "✅ Analysis Complete."
        return None, None, None, "❌ Error parsing files."
    raise PreventUpdate

@app.callback(
    Output('dashboard-content', 'style'),
    Output('signal-checklist', 'options'),
    Output('gps-map', 'figure'),
    Output('kpi-row', 'children'),
    Output('behavior-analysis-output', 'children'),
    Output('uds-auth-output', 'children'),
    Input('processed-data-store', 'data'),
    State('active-signals-store', 'data'),
    State('dbc-store', 'data')
)
def show_dashboard_and_initialize(jsonified_data, active_signals, dbc_data):
    if not jsonified_data: return {'display': 'none'}, [], go.Figure(), [], "", ""

    try:
        options = [{'label': s, 'value': s} for s in active_signals] if active_signals else []
        df = pd.read_json(io.StringIO(jsonified_data), orient='split')
        df.index = pd.to_datetime(df.index)

        fig_map, kpi_children = create_map_and_kpis(df, active_signals, None)
        
        behavior_analysis = analyze_vehicle_behavior_with_dbc(df, dbc_data) if dbc_data else "No DBC data"
        # Directly use the combined DataFrame which inherently contains 'can_id' and 'can_data'
        uds_analysis = check_uds_authentication_sequence(df)

        return (
            {'display': 'block'}, options, fig_map, kpi_children,
            html.Pre(behavior_analysis, style={'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace'}),
            html.Pre(uds_analysis, style={'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace'})
        )
    except Exception as e:
        print(f"Error initializing: {e}")
        return {'display': 'none'}, [], go.Figure(), [], str(e), str(e)

@app.callback(
    Output('graphs-container', 'children'),
    Output('hover-time-store', 'data'),
    Input('plot-button', 'n_clicks'),
    State('signal-checklist', 'value'),
    State('processed-data-store', 'data'),
    prevent_initial_call=True
)
def render_graphs(n_clicks, selected_signals, jsonified_data):
    if not jsonified_data or not selected_signals or n_clicks == 0: return [], None
    try:
        df = pd.read_json(io.StringIO(jsonified_data), orient='split')
        df.index = pd.to_datetime(df.index)
        graphs = []
        for i, signal in enumerate(selected_signals):
            downsampled_df = downsample_data(df, [signal])
            if not downsampled_df.empty:
                fig = go.Figure(go.Scattergl(x=downsampled_df.index, y=downsampled_df[signal], mode='lines', name=signal))
                fig.update_layout(title=signal, template='plotly_dark', height=300, margin=dict(l=40, r=20, t=40, b=30))
                graphs.append(dcc.Graph(id={'type': 'dynamic-graph', 'index': i}, figure=fig, style={'marginBottom': '10px'}))
        return graphs, None
    except Exception as e:
        print(f"Graph error: {e}")
        return [], None

@app.callback(
    Output('hover-time-store', 'data', allow_duplicate=True),
    Input({'type': 'dynamic-graph', 'index': dash.ALL}, 'hoverData'),
    prevent_initial_call=True
)
def update_hover_time(hoverData_list):
    if not hoverData_list or all(d is None for d in hoverData_list): raise PreventUpdate
    hoverData = next((item for item in hoverData_list if item is not None), None)
    if hoverData and 'points' in hoverData: return hoverData['points'][0]['x']
    raise PreventUpdate

@app.callback(
    Output('gps-map', 'figure', allow_duplicate=True),
    Output('kpi-row', 'children', allow_duplicate=True),
    Input('hover-time-store', 'data'),
    State('processed-data-store', 'data'),
    State('active-signals-store', 'data'),
    prevent_initial_call=True
)
def update_map_kpis(hover_time, jsonified_data, active_signals):
    if not jsonified_data: raise PreventUpdate
    df = pd.read_json(io.StringIO(jsonified_data), orient='split')
    df.index = pd.to_datetime(df.index)
    return create_map_and_kpis(df, active_signals, hover_time)

if __name__ == '__main__':
    app.run(debug=True)
