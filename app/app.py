import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from darts import Dartboard

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

db = Dartboard(pixels=2001)

fig = px.imshow(db.db_score_map, origin='lower', width=1000, height=1000)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)

app.layout = html.Div(
    children=[
        html.H1(children='Dartboard'),
        html.Div(children='The dartboard visualised:'),
        dcc.Graph(id='dartboard', figure=fig)
    ],
)

if __name__ == '__main__':
    app.run_server(debug=True)
