from dash import dcc, html, dash_table
import plotly.graph_objs as go
import pandas as pd
import numpy as np

############ FORMATTING ############

titletext = {
    'color': '#FFFFFF',
    'font-family': 'Helvetica',
    'font-size': '36px',
    'textAlign': 'left'
}

dashboard_header = {
    'color': '#393939',
    'font-family': 'Helvetica',
    'font-size': '30px',
    'textAlign': 'center',
    'font-variant-caps': 'all-small-caps',
    'font-weight': 'bold',
    'letter-spacing': '1px',
    'height': '60px',
    'vertical-align': 'bottom'
}

navbarcurrentpage = {
    'color': '#FFFFFF',
    'font-family': 'Helvetica',
    'font-size': '15px',
    'font-weight': 'bold',
    'textAlign': 'center',
    'line-height': '24px',
    'vertical-align': 'bottom'
}

navbarotherpage = {
    'color': '#FFFFFF',
    'font-family': 'Helvetica',
    'font-weight': 'light',
    'font-size': '15px',
    'textAlign': 'center',
    'line-height': '24px'
}

userinputinstr = {
    'color': '#525252',
    'font-family': 'Helvetica',
    'font-size': '12px',
    'font-weight': 'bold',
    'textAlign': 'right',
    'padding-left': '0px'
}

userinputtext = {
    'color': '#525252',
    'font-family': 'Helvetica',
    'font-weight': 'light',
    'font-size': '12px',
    'textAlign': 'center',
    'borderWidth': '0 0 0.5px 0',
    'background-color': '#f4f4f4'
}

filterinstr = {
    'color': '#525252',
    'font-family': 'Helvetica',
    'font-size': '12px',
    'font-weight': 'bold',
    'textAlign': 'center'
}

directioninstr = {
    'color': '#525252',
    'font-family': 'Helvetica',
    'font-size': '14px',
    'textAlign': 'center'
}

filtertext = {
    'color': '#525252',
    'font-family': 'Helvetica',
    'font-weight': 'light',
    'font-size': '12px',
    'textAlign': 'center',
    'borderWidth': '0 0 0.5px 0',
}

copy_header = {
    'color': '#525252',
    'font-family': 'Helvetica',
    'letter-spacing': '1.5px',
    'font-size': '20px',
    'textAlign': 'left',
    'padding-left': '0px'
}

copy_text = {
    'color': '#525252',
    'font-family': 'Helvetica',
    'font-weight': 'bold',
    'font-size': '20px',
    'padding-top': '10px',
    'textAlign': 'left',
    'borderWidth': '0 0 0.5px 0',
    'background-color': '#f4f4f4'
}

filter_instr_header = {
    'color': '#525252',
    'font-family': 'Helvetica',
    'font-weight': 'bold',
    'font-size': '20px',
    'padding-top': '20px',
    'padding-bottom': '20px',
    'textAlign': 'left',
    'borderWidth': '0 0 0.5px 0',
}

copy_text_l = {
    'color': '#525252',
    'font-family': 'Helvetica',
    'font-weight': 'lighter',
    'font-size': '20px',
    'padding-top': '10px',
    'textAlign': 'left',
    'borderWidth': '0 0 0.5px 0',
    'background-color': '#f4f4f4'
}

filter_headers = {
    'color': '#525252',
    'font-family': 'Helvetica',
    'letter-spacing': '1.5px',
    'font-size': '20px',
    'textAlign': 'left',
    'padding-left': '0px',
    'padding-top': '45px'
}

filter_headers_dark = {
    'color': '#FFFFFF',
    'font-family': 'Helvetica',
    'letter-spacing': '1.5px',
    'font-size': '20px',
    'textAlign': 'left',
    'padding-left': '0px',
    'padding-top': '15px',
    'padding-bottom': '15px'
}

recipe_colors = {
    'pink-red': 'rgb(255, 101, 131)',
    'white': 'rgb(251, 251, 252)',
    'light-grey': 'rgb(208, 206, 206)',
    'dark-blue-grey': 'rgb(62, 64, 76)',
    'light blue': 'rgb(150, 174, 208)',
    'lily-scent-green': 'rgb(226,231,187)',
    'pine-glade': 'rgb(188,205,145)',
    'black': 'rgb(0,0,0)'
}

externalgraph_rowstyling = {
    'background-color': '#f4f4f4',
}

externalgraph_colstyling = {
    'background-color': '#f4f4f4',
    'padding-top': '15px'
}

############ DATA LOADING (ON PAGE OPEN) ############


# Load data
ingredients_filepath = 'data/user_ingredients.parquet'
techniques_filepath = 'data/user_techniques.parquet'
cuisines_filepath = 'data/user_cuisines.parquet'

live_ingr = pd.read_parquet(ingredients_filepath)
live_tech = pd.read_parquet(techniques_filepath)
live_cusi = pd.read_parquet(cuisines_filepath)

recipes_in_filepath = 'data/recipes_in_count2_mean4.parquet'
website_filepath = 'data/website.csv'
recipes_in = pd.read_parquet(recipes_in_filepath)
website = pd.read_csv(website_filepath, index_col=0)


# Generate random recipe table to pre-populate recipes
def create_random_recipe_table(rec_in, web):
    topn = (
        rec_in
            .sample(20)
            .join(web, how='inner')
    )
    topn['Recipe'] = "[" + topn['name'].apply(lambda x: x.title()) + "](" + topn['url'] + ")"
    topn['similarity_score'] = 'Random'
    return topn[['Recipe']]


recipes_out = create_random_recipe_table(recipes_in, website)


############ FUNCTIONS TO CREATE HEADER / NAVBAR / EMPTY ROW ############

# Define function to create the header at top of page, returning Div
def get_header():
    header = html.Div([
        html.Div([], className='col-1'),
        html.Div([
            html.H1(children='Recipe Exploration Tool for food.com',
                    style=titletext
                    )],
            className='col',
            style={'padding-top': '90px',
                   'padding-bottom': '1%',
                   'height': '150px'}
        )
    ],
        className='row',
        style={'background-color': '#393939',
               'vertical-align': 'bottom',
               'border-color': '#8d8d8d',
               'border-width': '0 0 0.5px 0',
               'border-style': 'solid'}
    )
    return header


# Define function to create the nav bar depending on the page
def get_navbar(p='summary'):
    # Page 1
    navbar_summary = html.Div([
        html.Div([], className='col-1', style={'background-color': '#393939'}),
        html.Div([
            dcc.Link(
                html.H4(children='User Summary',
                        style=navbarcurrentpage),
                href='/apps/User-Summary'
            )
        ],
            className='col-2',
            style={'background-color': '#4c4c4c',
                   'border-color': '#0f62fe',
                   'border-width': '0 0 0 6px',
                   'border-style': 'solid',
                   'width': '160px'}
        ),
        html.Div([
            dcc.Link(
                html.H4(children='Try New Recipes',
                        style=navbarotherpage),
                href='/apps/Try-New-Recipes'
            )
        ],
            className='col-2', style={'background-color': '#393939',
                                      'width': '160px'}
        ),
        html.Div([], className='col-7', style={'background-color': '#393939'})
    ],
        className='row',
        style={'padding-top': '0%',
               'padding-bottom': '0%'
               }
    )

    # Page 2
    navbar_new_recipe = html.Div([
        html.Div([], className='col-1', style={'background-color': '#393939'}),
        html.Div([
            dcc.Link(
                html.H4(children='User Summary',
                        style=navbarotherpage),
                href='/apps/User-Summary'
            )
        ],
            className='col-2',
            style={'background-color': '#393939',
                   'width': '160px'}
        ),
        html.Div([
            dcc.Link(
                html.H4(children='Try New Recipes',
                        style=navbarcurrentpage),
                href='/apps/Try-New-Recipes'
            )
        ],
            className='col-2',
            style={'background-color': '#4c4c4c',
                   'border-color': '#0f62fe',
                   'border-width': '0 0 0 6px',
                   'border-style': 'solid',
                   'width': '160px'}
        ),
        html.Div([], className='col-7', style={'background-color': '#393939'})
    ],
        className='row',
        style={'padding-top': '0%',
               'padding-bottom': '0%'}
    )

    if p == 'summary':
        return navbar_summary
    elif p == 'Try-New-Recipes':
        return navbar_new_recipe


# Define function to add a 40 pixel-high empty row
def get_emptyrow(h='40px'):
    emptyrow = html.Div([
        html.Div([
            html.Br()
        ], className='col')
    ],
        className='row',
        style={'height': h})
    return emptyrow


############ FIRST PAGE - USER SUMMARY ############

summary = html.Div([
    # Row 1 : Header
    get_header(),

    # Row 2 : Nav bar
    get_navbar('summary'),

    # Row 3 : User ID
    html.Div([  # Top-level row for user ID
        html.Div([], className='col-1'),   # Blank column
        html.Div([  # User ID text column (instructions for filter)
            html.H5(
                children='User ID:',
                style=userinputinstr
            ),
        ],
            style={'padding-right': '5px'},
            className='col-1'),
        html.Div([  # User ID input field column
            dcc.Input(id='user_id_input',
                      type="number",
                      debounce=True,
                      placeholder="input your user ID",
                      value=788,
                      style=userinputtext),
        ],
            style={'padding-left': '0px'},
            className='col-2'),
        html.Div([], className='col-8')  # Blank columns to finish the row
    ],
        style={'background-color': '#f4f4f4',
               'padding-top': '0px'},
        className='row'
    ),

    # Row 4 - Charts
    html.Div([  # Top-level row for chart objects
        html.Div([], id='dummy_2', className='col-1'),  # Blank 1 column
        html.Div([  # 10-column div for all objects
            html.Div([  # Row div for ingredients
                html.Div([  # Text column
                     html.H5(children='INGREDIENTS',
                             style=copy_header),
                     html.H5(children=r'''Do you cook with as many ingredients as some of our most creative contributors?''',
                             style=copy_text),
                     html.Br(),
                     html.H5(children=r'''Take a look at the left-hand chart and see! If you're interested in trying new techniques, click the 'Try New Recipes' tab above and follow the instructions there.''',
                             style=copy_text_l)
                     ],
                    className='col-4',
                    style={'padding-right': '80px'}
                ),
                html.Div([  # Graph column - ingredients graph
                    dcc.Graph(
                        id='ing_graph')
                ],
                    className='col-5'
                ),
                html.Div([  # Graph column - wordcloud
                    dcc.Graph(id='wordcloud_graph')
                ],
                    className='col-3'
                ),
            ],
                className='row'
            ),
            html.Br(),
            html.Div([  # Row div for techniques
                html.Div([  # Text column
                     html.H5(children='TECHNIQUES',
                             style=copy_header),
                     html.H5(children=r'''Have you tried as many different techniques as you'd like?''',
                             style=copy_text),
                     html.Br(),
                     html.H5(children=r'''Take a look at the histogram and see if you've tried as many different techniques in the kitchen as our typical user! If you're interested in trying new techniques, click the 'Try New Recipes' tab above and follow the instructions there.'Try New Recipes' tab above and follow the instructions there.''',
                             style=copy_text_l)
                ],
                    className='col-4',
                    style={'padding-right': '80px'}
                ),
                html.Div([  # Graph column
                    dcc.Graph(id='tech_graph')
                ],
                    className='col-8'
                ),
            ],
                className='row'
            ),
            html.Br(),
            html.Div([  # Row for cuisines
                html.Div([  # Text column
                     html.H5(children='CUISINES',
                             style=copy_header),
                     html.H5(children=r'''How often do you try different cuisines?''',
                             style=copy_text),
                     html.Br(),
                     html.H5(children=r'''Take a look at the chart on the left, and see if you've cooked as many different cuisines as our typical user! If you're interested in trying new cuisines, click the 'Try New Recipes' tab above and follow the instructions there.''',
                             style=copy_text_l)
                ],
                    className='col-4',
                    style={'padding-right': '80px'}
                ),
                html.Div([ # Graph column
                    dcc.Graph(id='cui_graph')
                ],
                    className='col-8'
                ),

            ],
                className='row'
            ),
            html.Br(),
            html.Div([  # Row for community objects
                html.Div([  # Text column
                     html.H5(children='COMMUNITY',
                             style=copy_header),
                     html.H5(children=r'''Are you just like others in the various communities of users at food.com?''',
                         style=copy_text),
                     html.Br(),
                     html.H5(children=r'''Take a look at the graph and see how you fit into the network of users on food.com! The darker circles represent communities which you're a member of. If you're interested in getting outside your bubble, click the 'Try New Recipes' tab above and follow the instructions there.''',
                         style=copy_text_l)
                ],
                    className='col-4',
                    style={'padding-right': '80px'}
                ),
                html.Div([  # Graph column
                    dcc.Graph(
                        id='network_graph')
                ],
                    className='col-8'
                ),
            ],
                className='row'
            ),
        ],
            className='col-10',
            style=externalgraph_colstyling
        ),
        html.Div([], className='col-1', id='dummy_1'),  # Blank 1 column (right-hand side
    ],
        className='row',
        style=externalgraph_rowstyling
    ),
])


############ SECOND PAGE - TRY NEW RECIPES ############

new_recipe = html.Div([
    # Row 1 : Header
    get_header(),

    # Row 2 : Nav bar
    get_navbar('Try-New-Recipes'),

    # Row 3 : User ID
    html.Div([  # Top-level row for user ID
        html.Div([], className='col-1'),  # Blank column
        html.Div([  # User ID text column (instructions for filter)
            html.H5(
                children='User ID:',
                style=userinputinstr
            ),
        ],
            style={'padding-right': '5px'},
            className='col-1'),
        html.Div([  # User ID input field column
            dcc.Input(id='user_id_input',
                      type="number",
                      debounce=True,
                      placeholder="input your user ID",
                      value=788,
                      style=userinputtext),
        ],
            style={'padding-left': '0px'},
            className='col-2'),
        html.Div([], className='col-8')  # Blank columns to finish the row
    ],
        style={'background-color': '#f4f4f4',
               'padding-top': '0px'},
        className='row'
    ),

    # Row 4 : Nutrition Filters
    html.Div([  # External row
        html.Div([], className='col-1'),  # 1 blank column
        html.Div([  # External 10-column
            html.Div(id='boundaries_title',
                     children='(1) SET SOME BOUNDARIES',
                     className='row',
                     style=filter_headers),
            html.Div(children='Select filters for any nutrition limitations',
                     className='row',
                     style=filter_instr_header),
            html.Div([
                html.Div([
                    html.H5('Calories', style=filterinstr),
                    dcc.Dropdown(id='calories-dropdown',
                                 options=[{"label": "< "+str(v), "value": v} for v in [100, 200, 300, 400, 500, 750, 1000, 2000, 5000, 10000, 90000]],
                                 value=None,
                                 style=filtertext
                                 )
                ],
                    className='col'
                ),
                html.Div([
                    html.H5('Carbs (g)', style=filterinstr),
                    dcc.Dropdown(id='carbs-dropdown',
                                 options=[{"label": "< "+str(v), "value": v} for v in [5, 10, 15, 25, 50, 100, 200, 500, 1000, 99999]],
                                 value=None,
                                 style=filtertext
                                 )
                ],
                    className='col'
                ),
                html.Div([
                    html.H5('Sugar (g)', style=filterinstr),
                    dcc.Dropdown(id='sugar-dropdown',
                                 options=[{"label": "< "+str(v), "value": v} for v in [5, 10, 15, 25, 50, 100, 200, 500, 1000, 999999]],
                                 value=None,
                                 style=filtertext
                                 )
                ],
                    className='col'
                ),
                html.Div([
                    html.H5('Saturated Fat (%DV)', style=filterinstr),
                    dcc.Dropdown(id='satfat-dropdown',
                                 options=[{"label": "< "+str(v), "value": v} for v in [5, 10, 15, 25, 50, 100, 200, 500, 1000, 99999]],
                                 value=None,
                                 style=filtertext
                                 )
                ],
                    className='col'
                ),
                html.Div([
                    html.H5('Protein (%DV)', style=filterinstr),
                    dcc.Dropdown(id='protein-dropdown',
                                 options=[{"label": "< "+str(v), "value": v} for v in [5, 10, 15, 25, 50, 100, 200, 500, 1000, 99999]],
                                 value=None,
                                 style=filtertext
                                 )
                ],
                    className='col')
            ],
                className='row'
            ),
            html.Div([], className='row', style={'height': '30px'}), # Blank row
            html.Div([
                html.Div([
                    html.Div(children='Select filter for the total cook time',
                             className='row',
                             style=filter_instr_header),
                    html.Div([
                        html.H5('Minutes to cook', style=filterinstr),
                        dcc.Dropdown(id='minute-dropdown',
                                     options=[{"label": "< "+str(v), "value": v} for v in [15, 30, 45, 60, 90, 120, 180, 360, 999999]],
                                     value=None,
                                     style=filtertext
                                     )
                    ],
                        className='col-3'
                    )
                ],
                    className='col-6'
                ),
                html.Div([
                    html.Div(children='Select filter for the type of meal',
                             className='row',
                             style=filter_instr_header),
                    html.Div([
                        html.H5('Meal', style=filterinstr),
                        dcc.Dropdown(id='meal-dropdown',
                                     options=['Side Dishes', 'Breakfast', 'Main Dish', 'Beverages',
                                        'Appetizers', 'Desserts', 'Lunch', 'Brunch', 'Snacks'],
                                     value=None,
                                     multi=True,
                                     style={
                                        'color': '#525252',
                                        'font-family': 'Helvetica',
                                        'font-weight': 'light',
                                        'font-size': '12px',
                                        'textAlign': 'left',
                                        'borderWidth': '0 0 0.5px 0',
                                        }
                                     )
                    ],
                        className='col-6'
                    )
                ],
                    className='col-6'
                )
            ],
                className='row'
            )
        ],
            className='col'
        ),
        html.Div([], className='col-1'),  # 1 blank column
    ],
        className='row',
        style=externalgraph_rowstyling
    ),

    # Row 5 - Blank
    get_emptyrow(),

    # Row 6 : Recommendation Filters
    html.Div([  # External row
        html.Div([], className='col-1'),  # 1 blank column
        html.Div([ # Column for similarity sliders
            html.Div(children='(2) PICK A DIRECTION',
                     className='row',
                     style=filter_headers),
            html.Div(children='Provide feedback to inform recommendations',
                     className='row',
                     style=filter_instr_header),
            html.Div([
                html.Div([
                    html.H5("I'd like recommended recipes' ingredients to be...", style=directioninstr),
                    dcc.Slider(-1, 1, value=0, included=False,
                               id='ingr-slider',
                               marks={
                                   -1: {'label': 'Different'},
                                   0: {'label': "Doesn't Matter"},
                                   1: {'label': 'Similar'}
                               })
                ],
                    className='col'
                ),
                html.Div([
                    html.H5("I'd like recommended recipes' techniques to be...", style=directioninstr),
                    dcc.Slider(-1, 1, value=0, included=False,
                               id='techniq_slider',
                               marks={
                                   -1: {'label': 'Different'},
                                   0: {'label': "Doesn't Matter"},
                                   1: {'label': 'Similar'}
                               })
                ],
                    className='col'
                ),
                html.Div([
                    html.H5("I'd like recommended recipes' cuisines to be...", style=directioninstr),
                    dcc.Slider(-1, 1, value=0, included=False,
                               id='cuisine-slider',
                               marks={
                                   -1: {'label': 'Different'},
                                   0: {'label': "Doesn't Matter"},
                                   1: {'label': 'Similar'}
                               })
                ],
                    className='col'
                ),
                html.Div([
                    html.H5("I'd like recommended recipes' user communities to be...", style=directioninstr),
                    dcc.Slider(-1, 1, value=0, included=False,
                               id='community-slider',
                               marks={
                                   -1: {'label': 'Different'},
                                   0: {'label': "Doesn't Matter"},
                                   1: {'label': 'Similar'}
                               })
                ],
                    className='col'
                )
            ],
                className='row'
            )
        ],
            className='col-10'
        ),
        html.Div([], className='col-1'),  # 1 blank column
    ],
        className='row'
    ),

    # Row 7 - Blank
    get_emptyrow(),

    # Row 8 - Generate Recs
    html.Div([
        html.Div([], className='col-1'),
        html.Div([
            html.Div(children='(3) GENERATE RECS!',
                     className='row',
                     style=filter_headers),
            html.Div([
                html.Div([
                    html.H5('Generate Recipe Recommendations',
                            style=filter_instr_header),
                    
                    html.Button('Submit',
                                id='gen-recipes'
                                ),
                    html.H6('Hit submit and scroll down to see your recommendations!',
                        style={
                            'color': '#525252',
                            'font-family': 'Helvetica',
                            'font-size': '14px',
                            'textAlign': 'left'
                        }),
                    html.H6('Note: if "Doesn\'t Matter" is selected for all Directions above, output will be random recipes',
                        style={
                            'color': '#525252',
                            'font-family': 'Helvetica',
                            'font-size': '14px',
                            'textAlign': 'left'
                        }),
                ],
                    className='col'
                )
            ],
                className='row'
            )
        ],
            className='col-10'
        ),
        html.Div([], className='col-1')
    ],
        className='row'
    ),

    # Row 9 - Blank
    get_emptyrow(),

    # Row 10 - Recipe recommendation list
    html.Div([
        html.Div([], className='col-1'),
        html.Div([
            html.Div(children='TOP RECIPE RECOMMENDATIONS',
                     id = 'top-recipes-recs-title',
                     className='row',
                     style=filter_headers_dark),
            dash_table.DataTable(id='rec-table',
                                 data=recipes_out.to_dict('records'),
                                 columns=[{"name": i, "id": i, 'type': 'text', 'presentation': 'markdown'} for i
                                          in recipes_out.columns],
                                 page_action='none',
                                 style_table={'height': '300px', 'overflowY': 'auto'},

                                 style_header={
                                     'backgroundColor': '#161616',
                                     'borderTop': '0',
                                     'borderBottom': '2px solid white',
                                     'borderLeft': '0',
                                     'borderRight': '0',
                                     'color': '#FFFFFF',
                                     'textAlign': 'left',
                                     'font-family': 'Helvetica',
                                     'font-weight': 'bold',
                                     'font-size': '16px'
                                 },
                                 style_data={
                                     'backgroundColor': '#161616',
                                     'borderBottom': '1px solid white',
                                     'borderLeft': '0',
                                     'borderRight': '0',
                                     'textAlign': 'left',
                                     'font-family': 'Helvetica',
                                     'font-size': '16px',
                                     'line-height': '30px',
                                 },
                                 )
        ],
            className='col-10'
        ),
        html.Div([], className='col-1'),
    ],
        className='row',
        id='recipe_rec_row',
        style={'background-color': '#161616', 'display':'none'}
    ),

    # Row 11 - Blank
    get_emptyrow(),

    # Row 12 - Visualizations
    html.Div([
        html.Div([], className='col-1'),
        html.Div([
            html.Div([
                html.Div([
                    html.H5(id='ing_title',
                            children='INGREDIENTS',
                            style=copy_header),
                    html.H5(children=r'''If you wanted to explore new ingredients, you should see different ingredients in each wordcloud!''',
                            style=copy_text),
                ],
                    className='col-4'
                ),
                html.Div([
                    dcc.Graph(id='wordcloud_graph_user')
                ],
                    className='col-4'
                ),
                html.Div([
                    dcc.Graph(id='wordcloud_graph_recs')
                ],
                    className='col-4'
                )
            ],
                className='row'
            ),
            html.Div([], className='row', style={'height': '30px', 'background-color': '#f4f4f4'}),
            html.Div([
                html.Div([
                    html.H5(children='TECHNIQUES',
                            style=copy_header),
                    html.H5(
                        children=r'''If you wanted to explore new techniques, you should see orange bars next to new techniques in the chart!''',
                        style=copy_text),
                ],
                    className='col-4'
                ),
                html.Div([
                    dcc.Graph(id='technique_graph_recs')
                ],
                    className='col-8'
                ),
            ],
                className='row'
            ),
            html.Div([], className='row', id='dummy_2', style={'height': '30px', 'background-color': '#f4f4f4'}),
            html.Div([
                html.Div([
                    html.H5(children='CUISINES',
                            style=copy_header),
                    html.H5(children=r'''If you wanted to explore new cuisines, you should see orange bars above new cuisines in the chart!''',
                            style=copy_text),
                ],
                    className='col-4'
                ),
                html.Div([
                    dcc.Graph(id='cuisine_graph_recs')
                ],
                    className='col-8'
                ),
            ],
                className='row'
            ),
            html.Div([], className='row', style={'height': '30px', 'background-color': '#f4f4f4'}),
            html.Div([
                html.Div([
                    html.H5(children='COMMUNITY',
                            style=copy_header),
                    html.H5(children=r'''If you wanted to explore recipes from communities far from your current, you should see an orange outline around new communities! Note that blue communities represent your current cooking communities.''',
                            style=copy_text),
                ],
                    className='col-4'
                ),
                html.Div([
                    dcc.Graph(id='network_graph_recs')
                ],
                    className='col-8'
                ),
            ],
                className='row'
            )
        ],
            className='col-10'
        ),
        html.Div([], className='col-1', id='dummy_1')
    ],
        id='rec_vis_row',
        className='row',
        style={'display': 'none'}
    ),
    html.Div([], className='row', style={'height': '30px', 'background-color': '#f4f4f4'})
])
