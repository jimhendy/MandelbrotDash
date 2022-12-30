import dash
import dash_core_components as dcc
import dash_html_components as html
import numba
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from loguru import logger
import os

RESOLUTION = (1_500, 1_500)
MAX_ITERATIONS = 1_500
NORM_FUNC = np.log

INITIAL_LIMITS = [-2.5, 1.5, -2, 2]

GL = True

app = dash.Dash("Mandelbrot", suppress_callback_exceptions=True)
app.title = "Mandelbrot"
server = app.server

def get_axis_limits(x_or_y, new_data):
    key = f"{x_or_y}axis.range"
    if key in new_data:
        return new_data[key]
    elif key + "[0]" in new_data:
        return new_data[key + "[0]"], new_data[key + "[1]"]
    else:
        return None


@app.callback(
    Output("graph", "figure"),
    [Input("graph", "relayoutData")],
    [State("graph", "figure")],
)
def on_zoom(new_data, current_figure):
    if not (isinstance(new_data, dict)):
        raise PreventUpdate

    x_lims = get_axis_limits("x", new_data)
    y_lims = get_axis_limits("y", new_data)

    if x_lims is None and y_lims is None:
        if (
            new_data.get("xaxis.autorange") is True
            and new_data.get("yaxis.autorange") is True
        ):
            # Reset
            return im_show(*INITIAL_LIMITS)
        else:
            raise PreventUpdate
    else:
        if x_lims is not None:
            x_min, x_max = x_lims
        else:
            x_min, x_max = current_figure["layout"]["xaxis"]["range"]

        if y_lims is not None:
            y_min, y_max = y_lims
        else:
            y_min, y_max = current_figure['layout']['yaxis']['range']
    logger.info("Calculation complete")
    return im_show(x_min, x_max, y_min, y_max)

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
                xtemp = z_x * z_x - z_y * z_y + c_x
                z_y = 2 * z_x * z_y + c_y
                z_x = xtemp
                if z_x * z_x + z_y * z_y > 4:
                    break
            if it:
                n_iterations[yi][xi] = it

    return x, y, n_iterations


def im_show(x_min, x_max, y_min, y_max):
    x, y, n_iterations = get_n_iterations(x_min, x_max, y_min, y_max)
    z = NORM_FUNC(n_iterations)
    fig = px.imshow(z, origin='lower', binary_string=True, zmin=z.min(), zmax=z.max(), x=x, y=y, binary_compression_level=8)
    fig.update_layout(
        margin = {'t': 0, 'b': 0, 'l': 0, 'r': 0},
        autosize=True,
        coloraxis_showscale=False,
        xaxis = {'showticklabels': False},
        yaxis = {'showticklabels': False},
        hovermode = False
    )
    return fig

app.layout = html.Div(
    id="main_div",
    children=dcc.Graph(
        id="graph",
        style={"width": "100%", "height": "100%"},
        figure=im_show(*INITIAL_LIMITS),
        config={'scrollZoom': True}
    ),
    style={
        "width": "calc(100vw - 16px)",
        "height": "calc(100vh - 16px)",
        "margin": {"t": 0, "b": 0, "l": 0, "r": 0},
    },
)

if __name__ == "__main__":
    server.run(debug=False, port=os.environ.get("PORT", 8080))
