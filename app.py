import dash
import dash_core_components as dcc
import dash_html_components as html
import numba
import numpy as np
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

RESOLUTION = (1_000, 1_000)
MAX_ITERATIONS = 2_000

INITIAL_LIMITS = [-2.5, 1.5, -2, 2]

GL = False

app = dash.Dash("Mandelbrot")


def get_axis_limits(x_or_y, new_data):
    key = f'{x_or_y}axis.range'
    if key in new_data:
        return new_data[key]
    elif key + '[0]' in new_data:
        return new_data[key+'[0]'], new_data[key+'[1]']
    else:
        return None


@app.callback(
    Output("graph", "figure"),
    [Input("graph", "relayoutData")],
    [State("graph", "figure")]
)
def on_zoom(new_data, current_figure):
    if not (isinstance(new_data, dict)):
        raise PreventUpdate

    x_lims = get_axis_limits('x', new_data)
    y_lims = get_axis_limits('y', new_data)

    if x_lims is None and y_lims is None:
        if new_data.get('xaxis.autorange') is True and new_data.get('yaxis.autorange') is True:
            # Reset
            return get_figure(*INITIAL_LIMITS)
        else:
            raise PreventUpdate
    else:
        if x_lims is not None:
            x_min, x_max = x_lims
        else:
            x_min, x_max = current_figure['layout']['xaxis']['range']

        if y_lims is not None:
            y_min, y_max = y_lims
        else:
            y_min, y_max = current_figure['layout']['yaxis']['range']

    return get_figure(x_min, x_max, y_min, y_max)


@numba.njit(parallel=True)
def get_n_iterations(x_min, x_max, y_min, y_max):

    x = np.linspace(x_min, x_max, RESOLUTION[0])
    y = np.linspace(y_min, y_max, RESOLUTION[1])

    n_iterations = np.ones((y.shape[0], x.shape[0]))

    for xi in numba.prange(x.shape[0]):
        c_x = x[xi]
        for yi in numba.prange(y.shape[0]):
            c_y = y[yi]
            z_x = 0
            z_y = 0
            for it in range(MAX_ITERATIONS):
                xtemp = z_x*z_x - z_y*z_y + c_x
                z_y = 2 * z_x * z_y + c_y
                z_x = xtemp
                if z_x*z_x + z_y*z_y > 4:
                    break
            if it:
                n_iterations[yi][xi] = it

    return x, y, n_iterations


def get_figure(x_min, x_max, y_min, y_max):
    x, y, n_iterations = get_n_iterations(x_min, x_max, y_min, y_max)
    hm = go.Heatmapgl if GL else go.Heatmap

    return go.Figure(
        data=[
            hm(
                z=np.log(n_iterations),
                x=x,
                y=y,
                showscale=False,
                hoverinfo='none',
                zsmooth=False
            )
        ],
        layout={
            'margin': {'t': 0, 'b': 0, 'l': 0, 'r': 0},
            'xaxis': {'showticklabels': False},
            'yaxis': {'showticklabels': False},
            "hovermode": False
        }
    )


app.layout = html.Div(
    id="main_div",
    children=dcc.Graph(
        id='graph',
        style={"width": "100%", "height": "100%"},
        figure=get_figure(*INITIAL_LIMITS),
        config={'scrollZoom': True}
    ),
    style={
        "width": "calc(100vw - 16px)",
        "height": "calc(100vh - 16px)",
        'margin': {'t': 0, 'b': 0, 'l': 0, 'r': 0},
    },
)

if __name__ == '__main__':
    app.run_server(debug=False, host="localhost")
