from dash import dcc
from dash import html
import dash
from app import app
from app import server
from layouts import summary, new_recipe
import callbacks

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])

def display_page(pathname):
    if pathname == '/apps/User-Summary':
         return summary
    elif pathname == '/apps/Try-New-Recipes':
         return new_recipe
    else:
        return summary


if __name__ == '__main__':
    app.run_server(debug=False)
