import numpy as np
import plotly.graph_objs as go


def draw_cost(value_array: np.ndarray, x_name: str, y_name: str, smth: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[i + 1 for i in range(len(value_array))], y=value_array))
    fig.update_layout(legend_orientation="h",
                      legend=dict(x=.5, xanchor="center"),
                      title=smth,
                      xaxis_title=x_name,
                      yaxis_title=y_name,
                      margin=dict(l=0, r=0, t=30, b=0))

    fig.show()
    return fig


def draw_model_real(model_out: np.ndarray, real_out: np.ndarray, name: str, x_axis_name: str, y_axis_name: str,
                    real_color: str, model_color: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[i for i in range(len(model_out))], y=model_out, name="model " + name,
                             line=dict(color=model_color), mode="markers"))
    fig.add_trace(go.Scatter(x=[i for i in range(len(model_out))], y=real_out, name="real " + name,
                             line=dict(color=real_color), mode="markers"))
    fig.update_traces(showlegend=True)
    fig.update_layout(legend_orientation="h", title=name, xaxis_title=x_axis_name, yaxis_title=y_axis_name)
    fig.show()
