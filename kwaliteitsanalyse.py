import requests, json, pandas as pd # type: ignore


def fetch_data(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()



url1 = "https://api.linkeddata.cultureelerfgoed.nl/queries/sablina-vis/topTerm/2/run?pageSize=10000"
url2 = "https://api.linkeddata.cultureelerfgoed.nl/queries/sablina-vis/missingscopenote01/18/run?pageSize=10000"
url3 = "https://api.linkeddata.cultureelerfgoed.nl/queries/sablina-vis/missingscopenote02/34/run?pageSize=10000"
url4 = "https://api.linkeddata.cultureelerfgoed.nl/queries/sablina-vis/missingscopenote03/35/run?pageSize=10000"
url5 = "https://api.linkeddata.cultureelerfgoed.nl/queries/sablina-vis/missingscopenote04/20/run?pageSize=10000"
url6 = "https://api.linkeddata.cultureelerfgoed.nl/queries/sablina-vis/missingscopenote05/17/run?pageSize=10000"
url7 = "https://api.linkeddata.cultureelerfgoed.nl/queries/sablina-vis/missingscopenote06/13/run?pageSize=10000"
url8 = "https://api.linkeddata.cultureelerfgoed.nl/queries/sablina-vis/missingscopenote07/15/run?pageSize=10000"
url9 = "https://api.linkeddata.cultureelerfgoed.nl/queries/sablina-vis/missingscopenote08/17/run?pageSize=10000"
url10 = "https://api.linkeddata.cultureelerfgoed.nl/queries/sablina-vis/missingscopenote09/14/run?pageSize=10000"
url11 = "https://api.linkeddata.cultureelerfgoed.nl/queries/sablina-vis/missingscopenote10/8/run?pageSize=10000"


urls = [url1, url2, url3, url4, url5, url6, url7, url8, url9, url10, url11]

for i, url in enumerate(urls, start=1):
    data = fetch_data(url)
    globals()[f"dataLaag{i}"] = data

    dfs = ['dataLaag1', 'dataLaag2', 'dataLaag3', 'dataLaag4', 'dataLaag5', 'dataLaag6', 'dataLaag7', 'dataLaag8', 'dataLaag9', 'dataLaag10', 'dataLaag11']



for i, df in enumerate(dfs, start=1):
    df = globals()[df]
    df = pd.DataFrame(df)
    globals()[f"df{i}"] = df



df1.rename(columns={'topTermLabel': 'prefLabelLaag1'}, inplace=True) # type: ignore
# df1.rename(columns={'countWithoutScopeNote': 'aantalNarrowerZonderScopeNote_df1'}, inplace=True) # type: ignore


dfs2 = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11]

# List of merge keys (one per merge step):
merge_keys = [
    'prefLabelLaag1',  # merge df1 and df2
    'prefLabelLaag2',  # merge (df1+df2) and df3
    'prefLabelLaag3',  # merge previous result and df4
    'prefLabelLaag4',  # merge with df5
    'prefLabelLaag5',  # merge with df6
    'prefLabelLaag6',  # merge with df7
    'prefLabelLaag7',  # merge with df8
    'prefLabelLaag8',  # merge with df9
    'prefLabelLaag9',  # merge with df10
    'prefLabelLaag10'  # merge with df11
]

merged_df = dfs2[0]

for i in range(len(merge_keys)):
    suffix_left = f"_df{i+1}"
    suffix_right = f"_df{i+2}"
    
    merged_df = pd.merge(
        merged_df,
        dfs2[i+1],
        on=merge_keys[i],
        how='outer',
        suffixes=(suffix_left, suffix_right)
    )

print(merged_df.head())



colsAantalNarrower = ['aantalNarrowerZonderScopeNote_df2','aantalNarrowerZonderScopeNote_df3','aantalNarrowerZonderScopeNote_df4','aantalNarrowerZonderScopeNote_df5',
                      'aantalNarrowerZonderScopeNote_df6','aantalNarrowerZonderScopeNote_df7','aantalNarrowerZonderScopeNote_df8','aantalNarrowerZonderScopeNote_df9',
                      'aantalNarrowerZonderScopeNote_df10']


merged_df[colsAantalNarrower] = merged_df[colsAantalNarrower].apply(pd.to_numeric, errors='coerce')



paired_columns = [
    ('mistEigenScopeNote_df2','aantalNarrowerZonderScopeNote_df2'),
('mistEigenScopeNote_df3','aantalNarrowerZonderScopeNote_df3'),
('mistEigenScopeNote_df4','aantalNarrowerZonderScopeNote_df4'),
('mistEigenScopeNote_df5','aantalNarrowerZonderScopeNote_df5'),
('mistEigenScopeNote_df6','aantalNarrowerZonderScopeNote_df6'),
('mistEigenScopeNote_df7','aantalNarrowerZonderScopeNote_df7'),
('mistEigenScopeNote_df8','aantalNarrowerZonderScopeNote_df8'),
('mistEigenScopeNote_df9','aantalNarrowerZonderScopeNote_df9'),
('mistEigenScopeNote_df10','aantalNarrowerZonderScopeNote_df10'),
('mistEigenScopeNote_df11','aantalNarrowerZonderScopeNote_df11')]


filtered_df = merged_df[
    ~merged_df.apply(
        lambda row: any(
            (row[pair[0]] == 0 and row[pair[1]] == 'nee') or
            (row[pair[1]] == 0 and row[pair[0]] == 'nee')
            for pair in paired_columns
        ),
        axis=1
    )
]




#try7
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np





max_level = 11  # total number of levels
level_columns = [f"prefLabelLaag{i}" for i in range(1, max_level+1)]

# =============================================================================
# the Dash App Layout
# =============================================================================
app = dash.Dash(__name__)

# Generate a list of dropdown containers for levels 1 to 11.
# The first dropdown is always visible; the others are initially hidden.
dropdown_divs = []

# Level 1 dropdown
dropdown_divs.append(
    html.Div([
        html.Label("Selecteer Top Concept:"),
        dcc.Dropdown(
            id="dropdown-1",
            options=[{'label': x, 'value': x} 
                     for x in sorted(filtered_df["prefLabelLaag1"].dropna().unique())],
            placeholder="Selecteer top concept"
        )
    ], style={'width': '30%', 'padding': '10px', 'display': 'block'})
)

# Dropdowns for levels 2 to 11 (initially hidden)
for i in range(2, max_level+1):
    dropdown_divs.append(
        html.Div([
            html.Label(f"Select prefLabelLaag{i}:"),
            dcc.Dropdown(
                id=f"dropdown-{i}",
                placeholder=f"Selecteer concept {i}"
            )
        ], id=f"div-dropdown-{i}", style={'width': '30%', 'padding': '10px', 'display': 'none'})
    )

app.layout = html.Div([
    html.H1("Dashboard Datakwaliteit Cultuurhistorische Thesaurus Concepten"),
    html.P("Selecteer een concept om het onderliggende concept weer te geven."),
    *dropdown_divs,
    html.H2("Samenvatting geselecteerde concept"),
    html.Div(id="summary-div"),
    html.H2("Tabel weergave"),
    dash_table.DataTable(
        id="data-table",
        columns=[{'name': col, 'id': col} for col in merged_df.columns],
        data=merged_df.to_dict("records"),
        page_size=10,
        filter_action="native",
        sort_action="native",
        style_table={'overflowX': 'auto'},
        style_cell={'whiteSpace': 'normal', 'textAlign': 'left'}
    )
], style={'padding': '20px'})

# =============================================================================
# Callback: Update Dropdown Options & Visibility for Levels 2 to 11
# =============================================================================
# This callback uses the current selections from all 11 dropdowns
# (values is a tuple of length 11: [dropdown-1 value, dropdown-2 value, ..., dropdown-11 value])
# For each level i (from 2 to 11), if all previous levels (1 to i-1) have a selection,
# then it filters merged_df accordingly, computes the unique values for level i, and makes its dropdown visible.
# Otherwise, the dropdown is hidden.
from dash.dependencies import Input, Output

# Create a list of inputs for dropdown-1 through dropdown-11.
input_ids = [Input(f"dropdown-{i}", "value") for i in range(1, max_level+1)]

# We need 2 outputs per dropdown for levels 2 to 11:
# one for the options and one for the style (to control visibility).
output_list = []
for i in range(2, max_level+1):
    output_list.append(Output(f"dropdown-{i}", "options"))
    output_list.append(Output(f"div-dropdown-{i}", "style"))

@app.callback(
    output_list,
    input_ids
)
def update_dropdowns(*vals):
    # vals is a tuple with 11 elements: (val1, val2, ..., val11)
    outputs = []
    for i in range(2, max_level+1):
        # For dropdown i, check if all previous levels (1 to i-1) are selected.
        valid = all(vals[j] is not None for j in range(i-1))
        if not valid:
            outputs.append([])  # options: empty list
            outputs.append({'width': '30%', 'padding': '10px', 'display': 'none'})
        else:
            # Filter the DataFrame using selections from levels 1 to i-1.
            df_filtered = merged_df.copy()
            for j in range(1, i):
                df_filtered = df_filtered[df_filtered[f"prefLabelLaag{j}"] == vals[j-1]]
            options = [{'label': x, 'value': x} 
                       for x in sorted(df_filtered[f"prefLabelLaag{i}"].dropna().unique())]
            outputs.append(options)
            outputs.append({'width': '30%', 'padding': '10px', 'display': 'block'})
    return outputs

# =============================================================================
# Callback: Update Summary Table Based on Selections (Deepest Level)
# =============================================================================
@app.callback(
    Output("summary-div", "children"),
    [Input(f"dropdown-{i}", "value") for i in range(1, max_level+1)]
)
def update_summary(*vals):
    # Find the deepest level for which a value is selected.
    deepest = 0
    for i, v in enumerate(vals, start=1):
        if v is not None:
            deepest = i
        else:
            break
    if deepest == 0:
        return "Geen selectie gemaakt."
    df_filtered = filtered_df.copy()
    for i in range(1, deepest+1):
        df_filtered = df_filtered[df_filtered[f"prefLabelLaag{i}"] == vals[i-1]]
    # Use appropriate summary columns:
    if deepest == 1:
        summary_cols = ["prefLabelLaag1", "countWithoutScopeNote"]
    else:
        summary_cols = [f"prefLabelLaag{deepest}",
                        f"mistEigenScopeNote_df{deepest}",
                        f"aantalNarrowerZonderScopeNote_df{deepest}",
                        f"uitklappen_df{deepest}"]
    summary_df = df_filtered[summary_cols].head(1)
    if summary_df.empty:
        return "No details available for this selection."
    return dash_table.DataTable(
        columns=[{'name': col, 'id': col} for col in summary_df.columns],
        data=summary_df.to_dict("records"),
        style_table={"overflowX": "auto"},
        page_size=1
    )

# =============================================================================
# Callback: Update Detailed Data Table Based on Selections
# =============================================================================
@app.callback(
    Output("data-table", "data"),
    [Input(f"dropdown-{i}", "value") for i in range(1, max_level+1)]
)
def update_data_table(*vals):
    df_filtered = merged_df.copy()
    for i, v in enumerate(vals, start=1):
        if v is not None:
            df_filtered = df_filtered[df_filtered[f"prefLabelLaag{i}"] == v]
    return df_filtered.to_dict("records")

# =============================================================================
# Run the App
# =============================================================================
if __name__ == '__main__':
    app.run_server(debug=True)


