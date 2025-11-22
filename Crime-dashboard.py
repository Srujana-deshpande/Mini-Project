# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from dash import Dash, dcc, html, Input, Output
# import dash_bootstrap_components as dbc

# # =====================================================
# # 🔹 Load and Prepare Dataset
# # =====================================================
# df = pd.read_csv("districtwise-ipc-crimes-readable.csv")
# df = df.loc[:, ~df.columns.duplicated()]

# year_column = "Year"
# state_column = "State"
# district_column = "District"

# # =====================================================
# # 🔹 Urban/Rural Classification (Vectorized)
# # =====================================================
# df["District_lower"] = df["District"].str.lower().fillna("")

# urban_keywords = ["urban", "city", "metro", "metropolitan"]
# rural_keywords = ["rural"]
# major_urban_districts = [
#     "bengaluru", "bangalore", "mumbai", "delhi", "hyderabad", "chennai", "pune",
#     "kolkata", "ahmedabad", "kochi", "cochin", "jaipur", "lucknow", "patna",
#     "bhopal", "indore", "surat", "nagpur", "visakhapatnam", "vishakhapatnam",
#     "noida", "gurgaon", "chandigarh", "thane", "vadodara", "rajkot", "madurai"
# ]

# df["Area"] = "Rural"
# df.loc[df["District_lower"].str.contains("|".join(urban_keywords)), "Area"] = "Urban"
# df.loc[df["District_lower"].str.contains("|".join(major_urban_districts)), "Area"] = "Urban"
# df.loc[df["District_lower"].str.contains("|".join(rural_keywords)), "Area"] = "Rural"

# df.drop(columns=["District_lower"], inplace=True)

# # =====================================================
# # 🔹 Crime Columns
# # =====================================================
# crime_columns = [
#     c for c in df.select_dtypes(include=["int64", "float64"]).columns
#     if c not in ["Record ID", "State Code", "District Code", "Registration Circles", year_column]
# ]
# default_crime = crime_columns[0] if crime_columns else None

# # Precompute correlation for whole dataset
# corr = df[crime_columns].corr().round(2)

# # =====================================================
# # 🔹 Helper: empty figure with center message
# # =====================================================
# def empty_figure(message: str) -> go.Figure:
#     fig = go.Figure()
#     fig.add_annotation(
#         text=message,
#         xref="paper", yref="paper",
#         x=0.5, y=0.5,
#         showarrow=False,
#         font=dict(size=14)
#     )
#     fig.update_xaxes(visible=False)
#     fig.update_yaxes(visible=False)
#     fig.update_layout(template="plotly_white")
#     return fig

# # =====================================================
# # 🔹 Initialize App
# # =====================================================
# app = Dash(
#     __name__,
#     external_stylesheets=[dbc.themes.FLATLY],
#     title="Urban vs Rural Crime Patterns"
# )

# app.layout = dbc.Container(
#     fluid=True,
#     children=[
#         html.H2(
#             "🌆 Urban vs 🌾 Rural Crime Patterns in India",
#             style={"textAlign": "center", "marginTop": "20px", "marginBottom": "25px"}
#         ),

#         # Filters
#         dbc.Row([
#             dbc.Col([
#                 html.Label("Select Year:"),
#                 dcc.Dropdown(
#                     options=[{"label": int(y), "value": int(y)} for y in sorted(df[year_column].unique())],
#                     value=int(sorted(df[year_column].unique())[0]),
#                     id="year-dropdown",
#                     clearable=False,
#                 )
#             ], width=3),
#             dbc.Col([
#                 html.Label("Select State:"),
#                 dcc.Dropdown(
#                     options=[{"label": s, "value": s} for s in sorted(df[state_column].unique())],
#                     value=sorted(df[state_column].unique())[0],
#                     id="state-dropdown",
#                     clearable=False,
#                 )
#             ], width=5),
#             dbc.Col([
#                 html.Label("Select Crime Type:"),
#                 dcc.Dropdown(
#                     options=[{"label": c, "value": c} for c in sorted(crime_columns)],
#                     value=default_crime,
#                     id="crime-dropdown",
#                     clearable=False,
#                 )
#             ], width=4),
#         ], className="mb-4"),

#         html.Hr(),

#         # Urban
#         dbc.Row([
#             dbc.Col(dcc.Graph(id="urban-bar", style={"height": "400px"}), width=8),
#             dbc.Col(html.Div(id="urban-text", style={"padding": "20px"}), width=4),
#         ], className="mb-4"),

#         # Rural
#         dbc.Row([
#             dbc.Col(dcc.Graph(id="rural-bar", style={"height": "400px"}), width=8),
#             dbc.Col(html.Div(id="rural-text", style={"padding": "20px"}), width=4),
#         ], className="mb-4"),

#         # Trend
#         dbc.Row([
#             dbc.Col(dcc.Graph(id="urban-rural-trend", style={"height": "400px"}), width=8),
#             dbc.Col(html.Div(id="trend-text", style={"padding": "20px"}), width=4),
#         ], className="mb-4"),

#         # Pie
#         dbc.Row([
#             dbc.Col(dcc.Graph(id="crime-pie", style={"height": "400px"}), width=8),
#             dbc.Col(html.Div(id="pie-text", style={"padding": "20px"}), width=4),
#         ], className="mb-4"),

#         # Correlation
#         dbc.Row([
#             dbc.Col(dcc.Graph(id="crime-heatmap", style={"height": "400px"}), width=8),
#             dbc.Col(html.Div(id="corr-text", style={"padding": "20px"}), width=4),
#         ], className="mb-4"),

#         html.Hr(),

#         # Overall conclusion
#         html.Div(
#             id="overall-conclusion",
#             style={
#                 "padding": "25px",
#                 "background": "#eef5ff",
#                 "borderRadius": "10px",
#                 "marginTop": "10px",
#                 "fontSize": "18px",
#                 "fontWeight": "500"
#             }
#         ),
#     ]
# )

# # =====================================================
# # 🔹 Callback
# # =====================================================
# @app.callback(
#     [
#         Output("urban-bar", "figure"),
#         Output("urban-text", "children"),
#         Output("rural-bar", "figure"),
#         Output("rural-text", "children"),
#         Output("urban-rural-trend", "figure"),
#         Output("trend-text", "children"),
#         Output("crime-pie", "figure"),
#         Output("pie-text", "children"),
#         Output("crime-heatmap", "figure"),
#         Output("corr-text", "children"),
#         Output("overall-conclusion", "children"),
#     ],
#     [
#         Input("year-dropdown", "value"),
#         Input("state-dropdown", "value"),
#         Input("crime-dropdown", "value"),
#     ]
# )
# def update_graphs(selected_year, selected_state, selected_crime):
#     # Fallback crime column
#     if selected_crime not in df.columns:
#         selected_crime = default_crime

#     state_df = df[df[state_column] == selected_state]
#     filtered_df = state_df[state_df[year_column] == selected_year]

#     # If there is NO data at all for this state-year
#     if filtered_df.empty:
#         msg = f"No data available for {selected_state} in {selected_year}."
#         empty = empty_figure(msg)
#         fig_heat = px.imshow(corr, text_auto=True, title="Crime Correlation Heatmap")
#         for fig in (empty, fig_heat):
#             fig.update_layout(template="plotly_white", margin=dict(l=40, r=40, t=60, b=40))

#         conclusion = (
#             f"### 🔍 Overall Summary for {selected_state} ({selected_year})\n\n"
#             f"No crime records are available for this selection, so a detailed "
#             f"urban–rural comparison and category-wise breakdown cannot be generated. "
#             f"The correlation heatmap shown is based on the entire dataset, not just this state-year."
#         )

#         return (
#             empty, msg,
#             empty, msg,
#             empty, msg,
#             empty, msg,
#             fig_heat, "Correlation shown for full dataset (not filtered by state/year).",
#             conclusion,
#         )

#     # ---------- Urban ----------
#     urban_df = filtered_df[filtered_df["Area"] == "Urban"]
#     if urban_df.empty:
#         fig_urban = empty_figure("No urban districts classified for this selection.")
#         urban_text = (
#             f"For **{selected_state} ({selected_year})**, no districts are classified as Urban "
#             f"based on the current heuristic."
#         )
#         total_urban = 0
#     else:
#         top_urban = (
#             urban_df.groupby("District")[selected_crime]
#             .sum()
#             .nlargest(10)
#             .reset_index()
#         )
#         fig_urban = px.bar(
#             top_urban,
#             x="District",
#             y=selected_crime,
#             color="District",
#             title=f"Top 10 Urban Districts – {selected_crime}",
#         )
#         total_urban = int(urban_df[selected_crime].sum())
#         top_name = top_urban.iloc[0]["District"]
#         top_value = int(top_urban.iloc[0][selected_crime])
#         urban_text = (
#             f"Total reported **{selected_crime.lower()}** cases in urban districts: "
#             f"**{total_urban:,}**. The highest is in **{top_name}** with "
#             f"**{top_value:,}** cases."
#         )

#     # ---------- Rural ----------
#     rural_df = filtered_df[filtered_df["Area"] == "Rural"]
#     if rural_df.empty:
#         fig_rural = empty_figure("No rural districts classified for this selection.")
#         rural_text = (
#             f"For **{selected_state} ({selected_year})**, no districts are classified as Rural "
#             f"based on the current heuristic."
#         )
#         total_rural = 0
#     else:
#         top_rural = (
#             rural_df.groupby("District")[selected_crime]
#             .sum()
#             .nlargest(10)
#             .reset_index()
#         )
#         fig_rural = px.bar(
#             top_rural,
#             x="District",
#             y=selected_crime,
#             color="District",
#             title=f"Top 10 Rural Districts – {selected_crime}",
#         )
#         total_rural = int(rural_df[selected_crime].sum())
#         top_r_name = top_rural.iloc[0]["District"]
#         top_r_value = int(top_rural.iloc[0][selected_crime])
#         rural_text = (
#             f"Total reported **{selected_crime.lower()}** cases in rural districts: "
#             f"**{total_rural:,}**. The highest is in **{top_r_name}** with "
#             f"**{top_r_value:,}** cases."
#         )

#     # ---------- Trend ----------
#     if state_df.empty:
#         fig_trend = empty_figure("No data available for this state.")
#         trend_text = f"No trend data available for **{selected_state}**."
#     else:
#         trend_data = (
#             state_df.groupby([year_column, "Area"], as_index=False)[selected_crime]
#             .sum()
#         )
#         fig_trend = px.line(
#             trend_data,
#             x=year_column,
#             y=selected_crime,
#             color="Area",
#             markers=True,
#             title=f"Trend of {selected_crime} in Urban vs Rural Areas",
#         )
#         max_idx = trend_data[selected_crime].idxmax()
#         max_row = trend_data.loc[max_idx]
#         trend_text = (
#             f"The highest **{selected_crime.lower()}** count in **{selected_state}** "
#             f"occurs in **{int(max_row[year_column])}** in **{max_row['Area']}** areas "
#             f"with **{int(max_row[selected_crime]):,}** cases."
#         )

#     # ---------- Pie (crime distribution) ----------
#     pie_data = filtered_df[crime_columns].sum().sort_values(ascending=False)
#     pie_total = pie_data.sum()

#     if pie_total == 0:
#         fig_pie = empty_figure("No crime counts available to build distribution.")
#         pie_text = (
#             f"No crime distribution can be shown for **{selected_state} ({selected_year})** "
#             f"because all crime counts are zero or missing."
#         )
#         pie_summary = (
#             "No single crime category dominates because recorded values are zero or missing."
#         )
#         top_crime = None
#         share = 0.0
#     else:
#         pie_top8 = pie_data.head(8)
#         fig_pie = px.pie(
#             values=pie_top8.values,
#             names=pie_top8.index,
#             title=f"Top Crime Categories in {selected_state} ({selected_year})",
#         )
#         top_crime = pie_top8.index[0]
#         share = round(pie_top8.iloc[0] / pie_top8.sum() * 100, 1)
#         pie_text = (
#             f"The most dominant crime category in **{selected_state} ({selected_year})** is "
#             f"**{top_crime}**, contributing about **{share}%** of crimes among the top categories."
#         )
#         pie_summary = (
#             f"The most prominent crime category is **{top_crime}**, "
#             f"accounting for roughly **{share}%** of recorded cases."
#         )

#     # ---------- Correlation ----------
#     fig_heat = px.imshow(
#         corr, text_auto=True, title="Crime Correlation Heatmap"
#     )
#     corr_no_diag = corr.copy()
#     for col in corr_no_diag.columns:
#         corr_no_diag.loc[col, col] = None
#     s = corr_no_diag.unstack().dropna()

#     if s.empty:
#         corr_text = (
#             "The correlation matrix does not show any strong relationships between crime categories."
#         )
#         corr_summary = (
#             "No strong correlation between specific crime categories could be identified."
#         )
#     else:
#         pair = s.abs().idxmax()          # pair is (crime1, crime2)
#         value = s.loc[pair]
#         corr_text = (
#             f"The strongest correlation is between **{pair[0]}** and **{pair[1]}** "
#             f"(correlation ≈ **{value:.2f}**), suggesting that these crimes tend to vary together."
#         )
#         corr_summary = (
#             f"The strongest relationship is between **{pair[0]}** and **{pair[1]}** "
#             f"(correlation ≈ **{value:.2f}**)."
#         )

#     # ---------- Overall Conclusion ----------
#     total_state_crime = int(filtered_df[crime_columns].sum().sum())

#     if total_urban > total_rural:
#         area_msg = "Urban areas report **higher crime counts** than rural regions in this selection."
#     elif total_rural > total_urban:
#         area_msg = "Rural areas report **more crime overall** than urban regions in this selection."
#     else:
#         area_msg = "Urban and rural areas report **similar overall crime levels** in this selection."

#     conclusion_lines = [
#         f"### 🔍 Overall Summary for {selected_state} ({selected_year})",
#         "",
#         f"- Total reported crimes (all categories combined): **{total_state_crime:,}**.",
#         f"- {area_msg}",
#         f"- {pie_summary}",
#         f"- {corr_summary}",
#         "",
#         "These insights highlight whether crime is more concentrated in urban or rural "
#         "districts, which categories dominate the crime profile, and which offences tend "
#         "to occur together. This can guide targeted policing and policy interventions."
#     ]
#     conclusion = "\n".join(conclusion_lines)

#     # ---------- Style all figures ----------
#     for fig in (fig_urban, fig_rural, fig_trend, fig_pie, fig_heat):
#         fig.update_layout(template="plotly_white", margin=dict(l=40, r=40, t=60, b=40))

#     return (
#         fig_urban, urban_text,
#         fig_rural, rural_text,
#         fig_trend, trend_text,
#         fig_pie, pie_text,
#         fig_heat, corr_text,
#         conclusion,
#     )


# # =====================================================
# # 🔹 Run App
# # =====================================================
# if __name__ == "__main__":
#     app.run(debug=True)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# =====================================================
# 🔹 Load and Prepare Dataset
# =====================================================
df = pd.read_csv("districtwise-ipc-crimes-readable.csv")
df = df.loc[:, ~df.columns.duplicated()]

year_column = "Year"
state_column = "State"
district_column = "District"

# =====================================================
# 🔹 Urban/Rural Classification (Vectorized)
# =====================================================
df["District_lower"] = df["District"].str.lower().fillna("")

urban_keywords = ["urban", "city", "metro", "metropolitan"]
rural_keywords = ["rural"]
major_urban_districts = [
    "bengaluru", "bangalore", "mumbai", "delhi", "hyderabad", "chennai", "pune",
    "kolkata", "ahmedabad", "kochi", "cochin", "jaipur", "lucknow", "patna",
    "bhopal", "indore", "surat", "nagpur", "visakhapatnam", "vishakhapatnam",
    "noida", "gurgaon", "chandigarh", "thane", "vadodara", "rajkot", "madurai"
]

df["Area"] = "Rural"
df.loc[df["District_lower"].str.contains("|".join(urban_keywords)), "Area"] = "Urban"
df.loc[df["District_lower"].str.contains("|".join(major_urban_districts)), "Area"] = "Urban"
df.loc[df["District_lower"].str.contains("|".join(rural_keywords)), "Area"] = "Rural"

df.drop(columns=["District_lower"], inplace=True)

# =====================================================
# 🔹 Crime Columns
# =====================================================
crime_columns = [
    c for c in df.select_dtypes(include=["int64", "float64"]).columns
    if c not in ["Record ID", "State Code", "District Code", "Registration Circles", year_column]
]
default_crime = crime_columns[0] if crime_columns else None

# Precompute correlation for whole dataset
corr = df[crime_columns].corr().round(2)


# =====================================================
# 🔹 Helper: empty figure with center message
# =====================================================
def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14)
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template="plotly_white")
    return fig


# =====================================================
# 🔹 Initialize App
# =====================================================
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Urban vs Rural Crime Patterns"
)

app.layout = dbc.Container(
    fluid=True,
    children=[
        html.H2(
            "🌆 Urban vs 🌾 Rural Crime Patterns in India",
            style={"textAlign": "center", "marginTop": "20px", "marginBottom": "25px"}
        ),

        # Filters
        dbc.Row([
            dbc.Col([
                html.Label("Select Year:"),
                dcc.Dropdown(
                    options=[{"label": int(y), "value": int(y)} for y in sorted(df[year_column].unique())],
                    value=int(sorted(df[year_column].unique())[0]),
                    id="year-dropdown",
                    clearable=False,
                )
            ], width=3),
            dbc.Col([
                html.Label("Select State:"),
                dcc.Dropdown(
                    options=[{"label": s, "value": s} for s in sorted(df[state_column].unique())],
                    value=sorted(df[state_column].unique())[0],
                    id="state-dropdown",
                    clearable=False,
                )
            ], width=5),
            dbc.Col([
                html.Label("Select Crime Type:"),
                dcc.Dropdown(
                    options=[{"label": c, "value": c} for c in sorted(crime_columns)],
                    value=default_crime,
                    id="crime-dropdown",
                    clearable=False,
                )
            ], width=4),
        ], className="mb-4"),

        html.Hr(),

        # Urban
        dbc.Row([
            dbc.Col(dcc.Graph(id="urban-bar", style={"height": "400px"}), width=8),
            dbc.Col(html.Div(id="urban-text", style={"padding": "20px"}), width=4),
        ], className="mb-4"),

        # Rural
        dbc.Row([
            dbc.Col(dcc.Graph(id="rural-bar", style={"height": "400px"}), width=8),
            dbc.Col(html.Div(id="rural-text", style={"padding": "20px"}), width=4),
        ], className="mb-4"),

        # Trend
        dbc.Row([
            dbc.Col(dcc.Graph(id="urban-rural-trend", style={"height": "400px"}), width=8),
            dbc.Col(html.Div(id="trend-text", style={"padding": "20px"}), width=4),
        ], className="mb-4"),

        # Pie
        dbc.Row([
            dbc.Col(dcc.Graph(id="crime-pie", style={"height": "400px"}), width=8),
            dbc.Col(html.Div(id="pie-text", style={"padding": "20px"}), width=4),
        ], className="mb-4"),

        # Correlation
        dbc.Row([
            dbc.Col(dcc.Graph(id="crime-heatmap", style={"height": "400px"}), width=8),
            dbc.Col(html.Div(id="corr-text", style={"padding": "20px"}), width=4),
        ], className="mb-4"),

        html.Hr(),

        # Overall conclusion
        html.Div(
            id="overall-conclusion",
            style={
                "padding": "25px",
                "background": "#eef5ff",
                "borderRadius": "10px",
                "marginTop": "10px",
                "fontSize": "18px",
                "fontWeight": "500"
            }
        ),
    ]
)


# =====================================================
# 🔹 Callback
# =====================================================
@app.callback(
    [
        Output("urban-bar", "figure"),
        Output("urban-text", "children"),
        Output("rural-bar", "figure"),
        Output("rural-text", "children"),
        Output("urban-rural-trend", "figure"),
        Output("trend-text", "children"),
        Output("crime-pie", "figure"),
        Output("pie-text", "children"),
        Output("crime-heatmap", "figure"),
        Output("corr-text", "children"),
        Output("overall-conclusion", "children"),
    ],
    [
        Input("year-dropdown", "value"),
        Input("state-dropdown", "value"),
        Input("crime-dropdown", "value"),
    ]
)
def update_graphs(selected_year, selected_state, selected_crime):
    # Fallback crime column
    if selected_crime not in df.columns:
        selected_crime = default_crime

    state_df = df[df[state_column] == selected_state]
    filtered_df = state_df[state_df[year_column] == selected_year]

    # If there is NO data at all for this state-year
    if filtered_df.empty:
        msg = f"No data available for {selected_state} in {selected_year}."
        empty = empty_figure(msg)
        fig_heat = px.imshow(corr, text_auto=True, title="Crime Correlation Heatmap")
        for fig in (empty, fig_heat):
            fig.update_layout(template="plotly_white", margin=dict(l=40, r=40, t=60, b=40))

        conclusion_md = dcc.Markdown(
            f"""
### 🔍 Overall Summary for {selected_state} ({selected_year})

No crime records are available for this selection, so a detailed urban–rural comparison and
crime-wise analysis cannot be generated.  

The correlation heatmap shown is based on the **entire dataset**, not just this state and year.
"""
        )

        return (
            empty, msg,
            empty, msg,
            empty, msg,
            empty, msg,
            fig_heat, "Correlation shown for full dataset (not filtered by state/year).",
            conclusion_md,
        )

    # ---------- Urban ----------
    urban_df = filtered_df[filtered_df["Area"] == "Urban"]
    if urban_df.empty:
        fig_urban = empty_figure("No urban districts classified for this selection.")
        urban_text = (
            f"For **{selected_state} ({selected_year})**, no districts are classified as Urban "
            f"based on the current heuristic."
        )
        total_urban = 0
        top_urban_name = None
        top_urban_value = 0
    else:
        top_urban_df = (
            urban_df.groupby("District")[selected_crime]
            .sum()
            .nlargest(10)
            .reset_index()
        )
        fig_urban = px.bar(
            top_urban_df,
            x="District",
            y=selected_crime,
            color="District",
            title=f"Top 10 Urban Districts – {selected_crime}",
        )
        total_urban = int(urban_df[selected_crime].sum())
        top_urban_name = top_urban_df.iloc[0]["District"]
        top_urban_value = int(top_urban_df.iloc[0][selected_crime])
        urban_text = (
            f"Total reported **{selected_crime.lower()}** cases in urban districts: "
            f"**{total_urban:,}**. The highest is in **{top_urban_name}** with "
            f"**{top_urban_value:,}** cases."
        )

    # ---------- Rural ----------
    rural_df = filtered_df[filtered_df["Area"] == "Rural"]
    if rural_df.empty:
        fig_rural = empty_figure("No rural districts classified for this selection.")
        rural_text = (
            f"For **{selected_state} ({selected_year})**, no districts are classified as Rural "
            f"based on the current heuristic."
        )
        total_rural = 0
        top_rural_name = None
        top_rural_value = 0
    else:
        top_rural_df = (
            rural_df.groupby("District")[selected_crime]
            .sum()
            .nlargest(10)
            .reset_index()
        )
        fig_rural = px.bar(
            top_rural_df,
            x="District",
            y=selected_crime,
            color="District",
            title=f"Top 10 Rural Districts – {selected_crime}",
        )
        total_rural = int(rural_df[selected_crime].sum())
        top_rural_name = top_rural_df.iloc[0]["District"]
        top_rural_value = int(top_rural_df.iloc[0][selected_crime])
        rural_text = (
            f"Total reported **{selected_crime.lower()}** cases in rural districts: "
            f"**{total_rural:,}**. The highest is in **{top_rural_name}** with "
            f"**{top_rural_value:,}** cases."
        )

    # ---------- Trend ----------
    if state_df.empty:
        fig_trend = empty_figure("No data available for this state.")
        trend_text = f"No trend data available for **{selected_state}**."
    else:
        trend_data = (
            state_df.groupby([year_column, "Area"], as_index=False)[selected_crime]
            .sum()
        )
        fig_trend = px.line(
            trend_data,
            x=year_column,
            y=selected_crime,
            color="Area",
            markers=True,
            title=f"Trend of {selected_crime} in Urban vs Rural Areas",
        )
        max_idx = trend_data[selected_crime].idxmax()
        max_row = trend_data.loc[max_idx]
        trend_text = (
            f"The highest **{selected_crime.lower()}** count in **{selected_state}** "
            f"occurs in **{int(max_row[year_column])}** in **{max_row['Area']}** areas "
            f"with **{int(max_row[selected_crime]):,}** cases."
        )

    # ---------- Pie (crime distribution) ----------
    pie_data = filtered_df[crime_columns].sum().sort_values(ascending=False)
    pie_total = pie_data.sum()

    if pie_total == 0:
        fig_pie = empty_figure("No crime counts available to build distribution.")
        pie_text = (
            f"No crime distribution can be shown for **{selected_state} ({selected_year})** "
            f"because all crime counts are zero or missing."
        )
        pie_summary = (
            "No single crime category dominates because recorded values are zero or missing."
        )
        top_crime = None
        share = 0.0
    else:
        pie_top8 = pie_data.head(8)
        fig_pie = px.pie(
            values=pie_top8.values,
            names=pie_top8.index,
            title=f"Top Crime Categories in {selected_state} ({selected_year})",
        )
        top_crime = pie_top8.index[0]
        share = round(pie_top8.iloc[0] / pie_top8.sum() * 100, 1)
        pie_text = (
            f"The most dominant crime category in **{selected_state} ({selected_year})** is "
            f"**{top_crime}**, contributing about **{share}%** of crimes among the top categories."
        )
        pie_summary = (
            f"The most prominent crime category is **{top_crime}**, "
            f"accounting for roughly **{share}%** of recorded cases."
        )

    # ---------- Correlation ----------
    fig_heat = px.imshow(
        corr, text_auto=True, title="Crime Correlation Heatmap"
    )
    corr_no_diag = corr.copy()
    for col in corr_no_diag.columns:
        corr_no_diag.loc[col, col] = None
    s = corr_no_diag.unstack().dropna()

    if s.empty:
        corr_text = (
            "The correlation matrix does not show any strong relationships between crime categories."
        )
        corr_summary = (
            "No strong correlation between specific crime categories could be identified."
        )
        pair = ("N/A", "N/A")
        value = 0.0
    else:
        pair = s.abs().idxmax()          # (crime1, crime2)
        value = s.loc[pair]
        corr_text = (
            f"The strongest correlation is between **{pair[0]}** and **{pair[1]}** "
            f"(correlation ≈ **{value:.2f}**), suggesting that these crimes tend to vary together."
        )
        corr_summary = (
            f"The strongest relationship is between **{pair[0]}** and **{pair[1]}** "
            f"(correlation ≈ **{value:.2f}**)."
        )

    # ---------- Overall Conclusion (crime-specific) ----------
    total_state_crime = int(filtered_df[crime_columns].sum().sum())

    if total_urban == 0 and total_rural == 0:
        dominance_sentence = (
            f"For the crime **{selected_crime}**, no cases are recorded in either "
            f"urban or rural areas for **{selected_state} ({selected_year})**."
        )
    else:
        if total_urban > total_rural:
            dominant_area = "Urban"
            dominant_value = total_urban
            lesser_value = total_rural
        elif total_rural > total_urban:
            dominant_area = "Rural"
            dominant_value = total_rural
            lesser_value = total_urban
        else:
            dominant_area = "Urban & Rural"
            dominant_value = total_urban
            lesser_value = total_rural  # same
        if dominant_area == "Urban & Rural":
            dominance_sentence = (
                f"For the crime **{selected_crime}**, urban and rural areas report "
                f"**similar levels** of cases (**{dominant_value:,}** each) in "
                f"**{selected_state} ({selected_year})**."
            )
        else:
            dominance_sentence = (
                f"The crime **{selected_crime}** is more prevalent in "
                f"**{dominant_area} areas**, reporting **{dominant_value:,}** cases, "
                f"compared to **{lesser_value:,}** cases in the opposite region."
            )

    if total_urban > total_rural and top_urban_name:
        main_hotspot = f"Most affected urban district: **{top_urban_name}** ({top_urban_value:,} cases)."
    elif total_rural > total_urban and top_rural_name:
        main_hotspot = f"Most affected rural district: **{top_rural_name}** ({top_rural_value:,} cases)."
    else:
        # either equal or missing info
        if top_urban_name and top_rural_name:
            main_hotspot = (
                f"Key hotspots include **{top_urban_name}** (urban) and **{top_rural_name}** (rural)."
            )
        elif top_urban_name:
            main_hotspot = f"Key hotspot: **{top_urban_name}** (urban)."
        elif top_rural_name:
            main_hotspot = f"Key hotspot: **{top_rural_name}** (rural)."
        else:
            main_hotspot = "Specific district hotspots cannot be identified from the current data."

    conclusion_text = f"""
### 🔍 Overall Summary for {selected_state} — {selected_crime} ({selected_year})

- Total reported crimes (all categories combined): **{total_state_crime:,}**  
- {dominance_sentence}  
- {pie_summary}  
- {corr_summary}  
- {main_hotspot}

These patterns show whether **{selected_crime}** is concentrated more in urban or rural
regions and highlight the main districts and crime categories that may require
greater policy and policing focus.
"""
    conclusion_md = dcc.Markdown(conclusion_text)

    # ---------- Style all figures ----------
    for fig in (fig_urban, fig_rural, fig_trend, fig_pie, fig_heat):
        fig.update_layout(template="plotly_white", margin=dict(l=40, r=40, t=60, b=40))

    return (
        fig_urban, urban_text,
        fig_rural, rural_text,
        fig_trend, trend_text,
        fig_pie, pie_text,
        fig_heat, corr_text,
        conclusion_md,
    )


# =====================================================
# 🔹 Run App
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)
