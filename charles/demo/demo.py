import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import numpy as np
import h5py

from charles.demo import demo_logic

from plotly.subplots import make_subplots
import plotly.graph_objects as go

app = dash.Dash(__name__)

# init concats all data in demo_logic.py into singuar set
demo_logic.initialize()

app.layout = html.Div([
    html.H1("SuperDARN Radar Demo"),

    # -------------------------------------------------------
    # search options section
    # -------------------------------------------------------
    html.H2("Search Options"),

    # search mode time, segment, random
    html.Div([
        html.Label("Search Mode:"),
        dcc.RadioItems(
            id="search_mode",
            options=[
                {"label": "Time (Closest)", "value": "time"},
                {"label": "Segment (Beam+ID)", "value": "segment"},
                {"label": "Random", "value": "random"}
            ],
            value="time",
            inline=True
        )
    ], style={"marginBottom": "15px"}),

    # time div
    html.Div([
        html.Label("Select Date:"),
        dcc.DatePickerSingle(
            id='date_picker',
            date='1995-06-19',  # default
            display_format='YYYY-MM-DD'
        ),
        html.Br(),
        html.Label("Select Hour (0-23):"),
        dcc.Dropdown(
            id='hour_dropdown',
            options=[{"label": f"{h:02d}", "value": h} for h in range(24)],
            value=1,
            clearable=False,
            style={"width": "100px"}
        )
    ], id="time_div", style={"display": "none", "marginBottom": "15px"}),


    # segment div
    html.Div([
        html.Label("Segment ID:"),
        dcc.Input(
            id="segment_id_input",
            type="number",
            value=1,
            min=0,
            max=4500,
            step=1,
            style={"width": "120px"}
        )
    ], id="segment_div", style={"display": "none", "marginBottom": "15px"}),

    # beam div (separate from segment id)
    html.Div([
        html.Label("Select Beam:"),
        dcc.Dropdown(
            id="beam_dropdown",
            options=[{"label": str(b), "value": b} for b in range(16)],
            value=0,
            clearable=False,
            style={"width": "100px"}
        )
    ], id="beam_div", style={"display": "none", "marginBottom": "15px"}),


    # same beam? to stop it retrieving the same event viewed from different beams
    html.Div([
        html.Label("Restrict Search to Same Beam?"),
        dcc.Checklist(
            id="same_beam",
            options=[{"label": "Yes", "value": "yes"}],
            value=[],
            inline=True
        )
    ], style={"marginBottom": "15px"}),

    # time exclusion search setting
    html.Div([
        html.Label("Time Exclusion:"),
        dcc.Dropdown(
            id="time_exclude",
            options=[
                {"label": "None", "value": "none"},
                {"label": "1 Hour", "value": "1h"},
                {"label": "4 Hours", "value": "4h"},
                {"label": "12 Hours", "value": "12h"},
                {"label": "1 Day", "value": "1d"},
                {"label": "1 Month", "value": "1m"}
            ],
            value="none",
            clearable=False
        )
    ], style={"marginBottom": "15px", "width": "200px"}),

    # how many nearest neighbours to retrieve
    html.Div([
        html.Label("Number of Neighbors to Retrieve (k):"),
        dcc.Slider(
            id="k_slider",
            min=1,
            max=15,
            step=1,
            value=5,
            marks={i: str(i) for i in range(1,16)}
        )
    ], style={"marginBottom": "15px"}),

    # search button
    html.Button("Search", id="search_button", n_clicks=0),
    html.Hr(),

    # -------------------------------------------------------
    # display options section
    # -------------------------------------------------------
    html.H2("Display Options"),

    # slider for how many neighbours to plot
    html.Div([
        html.Label("How many neighbors to plot?"),
        dcc.Slider(
            id="plot_k_slider",
            min=1,
            max=5,
            step=1,
            value=1,
            marks={i: str(i) for i in range(1,6)}
        )
    ], style={"marginBottom": "15px"}),

    html.Hr(),

    # output
    html.Div(id="neighbor_output", style={"whiteSpace": "pre-wrap"}),
    dcc.Graph(id="comparison_plot")
])

@app.callback(
    [Output("time_div", "style"), Output("segment_div", "style"), Output("beam_div", "style")],  # Change beam_dropdown → beam_div
    Input("search_mode", "value")
)
def toggle_search_inputs(search_mode):
    if search_mode == "time":
        return ({"display": "block"}, {"display": "none"}, {"display": "block"})
    elif search_mode == "segment":
        return ({"display": "none"}, {"display": "block"}, {"display": "block"})  # Beam is now shown for segments too
    else:
        return ({"display": "none"}, {"display": "none"}, {"display": "none"})


# main callback
@app.callback(
    [Output("neighbor_output", "children"), Output("comparison_plot", "figure")],
    [Input("search_button", "n_clicks")],
    [
        State("search_mode", "value"),
        State("date_picker", "date"),
        State("hour_dropdown", "value"),
        State("beam_dropdown", "value"),
        State("segment_id_input", "value"),
        State("same_beam", "value"),
        State("time_exclude", "value"),
        State("k_slider", "value"),
        State("plot_k_slider", "value")
    ]
)
def run_search(n_clicks,
               search_mode,
               date_val,
               hour_val,
               beam_val,
               seg_id_val,
               same_beam_list,
               time_exclude,
               k_neighbors,
               plot_k_neighbors
              ):

    # prompt if search not clicked
    if not n_clicks:
        return ("Please click 'Search' to begin.", px.imshow(np.zeros((30, 75))).to_dict())

    # locate anchor
    if search_mode == "time":
        if not date_val or hour_val is None:
            return ("Please select date/hour!", px.imshow(np.zeros((30, 75))).to_dict())
        combined_time_str = f"{date_val}T{hour_val:02d}:00:00"
        anchor_idx = demo_logic.find_anchor_by_time(combined_time_str, beam_number=beam_val)

    elif search_mode == "segment":
        if beam_val is None or seg_id_val is None:
            return ("Please select a beam + segment ID!", px.imshow(np.zeros((30,75))).to_dict())
        seg_str = f"beam_{beam_val}_segment_{seg_id_val}"
        anchor_idx = demo_logic.find_anchor_by_segment_name(seg_str)
    else:
        anchor_idx = np.random.randint(0, len(demo_logic.segment_names))

    # retrive neighbours
    same_beam = ("yes" in same_beam_list)
    neighbor_idxs = demo_logic.filter_neighbors(
        anchor_idx=anchor_idx,
        same_beam=same_beam,
        time_exclude=time_exclude,
        k=k_neighbors
    )
    if not neighbor_idxs:
        return (f"No neighbors found for anchor idx={anchor_idx}.", px.imshow(np.zeros((30, 75))).to_dict())

    # build list output
    neighbor_text_lines = []
    for i, nn_idx in enumerate(neighbor_idxs):
        seg_name = demo_logic.segment_names[nn_idx]
        # determine subset & local index for neighbor index `nn_idx`
        if nn_idx < demo_logic.offsets["train_end"]:
            s_name, s_local = "train", nn_idx
        elif nn_idx < demo_logic.offsets["val_end"]:
            s_name, s_local = "val", nn_idx - demo_logic.offsets["train_end"]
        else:
            s_name, s_local = "test", nn_idx - demo_logic.offsets["val_end"]

        h5_path = demo_logic.DATASET_PATHS[s_name]

        with h5py.File(h5_path, "r") as hf:
            beam_num = hf[seg_name].attrs.get("beam_number", -1)
            start_t = hf[seg_name].attrs.get("start_time", "unknown")

        neighbor_text_lines.append(f"{i+1}. {seg_name} [beam={beam_num}, start={start_t}]")

    # retrieve anchor info
    anchor_seg_name = demo_logic.segment_names[anchor_idx]
    # determine subset & local index for anchor index `anchor_idx`
    if anchor_idx < demo_logic.offsets["train_end"]:
        a_subset, a_local = "train", anchor_idx
    elif anchor_idx < demo_logic.offsets["val_end"]:
        a_subset, a_local = "val", anchor_idx - demo_logic.offsets["train_end"]
    else:
        a_subset, a_local = "test", anchor_idx - demo_logic.offsets["val_end"]

    a_file = demo_logic.DATASET_PATHS[a_subset]

    with h5py.File(a_file, "r") as hf:
        a_beam = hf[anchor_seg_name].attrs.get("beam_number", -1)
        a_start = hf[anchor_seg_name].attrs.get("start_time", "unknown")

    lines = []
    lines.append(f"Anchor (idx={anchor_idx}): {anchor_seg_name}, [beam={a_beam}, start={a_start}]\n")
    lines.append("Nearest Neighbors:")
    lines.extend(neighbor_text_lines)
    neighbor_output_str = "\n".join(lines)

    # plot neighbours
    plot_k = min(plot_k_neighbors, len(neighbor_idxs))
    fig = make_subplots(rows=1, cols=plot_k+1,
                        subplot_titles=["Anchor"] + [f"NN {i+1}" for i in range(plot_k)])

    # more anchor data
    anchor_data = demo_logic.get_plot_for(anchor_idx, anchor_idx)
    anchor_unscaled = anchor_data["anchor_seg"][2]
    import numpy.ma as ma
    anchor_mask = ma.masked_where(anchor_unscaled == -9999, anchor_unscaled)
    anchor_filled = anchor_mask.filled(np.nan).T
    anchor_min, anchor_max = (0, 1)
    if anchor_mask.count() > 0:
        anchor_min = anchor_mask.min()
        anchor_max = anchor_mask.max()

    # collect global min/max
    global_min, global_max = anchor_min, anchor_max

    # gather neighbour data
    neighbors_unscaled_data = []
    for idx in neighbor_idxs[:plot_k]:
        data_dict = demo_logic.get_plot_for(idx, idx)
        arr = data_dict["anchor_seg"][2]
        mask2 = ma.masked_where(arr == -9999, arr)
        if mask2.count() > 0:
            mn = mask2.min()
            mx = mask2.max()
            global_min = min(global_min, mn)
            global_max = max(global_max, mx)
        neighbors_unscaled_data.append(mask2)

    # plot anch in position 1
    fig.add_trace(
        go.Heatmap(
            z=anchor_filled,
            colorscale="Viridis",
            zmin=global_min,
            zmax=global_max
        ), row=1, col=1
    )

    # plot neighbours in following positions
    col_idx = 2
    for n_mask in neighbors_unscaled_data:
        n_filled = n_mask.filled(np.nan).T
        fig.add_trace(
            go.Heatmap(
                z=n_filled,
                colorscale="Viridis",
                zmin=global_min,
                zmax=global_max
            ), row=1, col=col_idx
        )
        col_idx += 1

    fig.update_layout(height=600, width=300*(plot_k+1),
                      title="Anchor vs. Neighbors")

    return (neighbor_output_str, fig)


if __name__ == "__main__":
    app.run_server(debug=True)
