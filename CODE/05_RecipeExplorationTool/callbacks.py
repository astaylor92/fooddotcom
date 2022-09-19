import matplotlib
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import dash
from app import app
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from time import time
from dash.dependencies import Output, Input

matplotlib.use('agg')  # set the backend of matplotlib plot to a non-interactive one

############ LOAD DATA #############
live_ingr = pd.read_parquet('data/user_ingredients.parquet')
live_ingr_ct = pd.read_parquet('data/user_ing_count.parquet')
live_tech = pd.read_parquet('data/user_techniques.parquet')
live_tech['techniques'] = live_tech['techniques'].apply(lambda x: x.title())
live_cusi = pd.read_parquet('data/user_cuisines.parquet')
live_cusi['cuisine'] = live_cusi['cuisine'].apply(lambda x: x.title())
recipes_in = pd.read_parquet('data/recipes_in_count2_mean4.parquet')
website = pd.read_csv('data/website.csv', index_col=0)
edges = pd.read_parquet('data/community_graph_edges_pctthresh-08_split.parquet')
users_in = pd.read_csv('data/users_in_count2_mean4.csv')
recipe2clust = pd.read_parquet('data/community_assignments_pctthresh-08_split_filled_filter.parquet')
clust_dist = pd.read_parquet('data/community_distances_pctthresh-08_split_adj.parquet')
ingr_vec = pd.read_parquet('data/recipes_ingredient_vec.parquet')
nutrition = pd.read_csv('data/nutrition.csv')

# Initialize total_clicks
total_clicks = 0

############# FUNCTIONS FOR TRY NEW RECIPES #########
# Create recipe table
def create_recipe_table(rec_in, web, sim):
    topn = (
        sim
        .sort_values('similarity_score',ascending=False)
        .iloc[:20,:]
        .join(rec_in, how='inner')
        .join(web, how='inner')
        )
    topn['Recipe'] = "["+topn['name']+"]("+topn['url']+")"
    return topn

# Generate random recipe table to pre-populate recipes
def create_random_recipe_table(rec_in, web):
    topn = (
        rec_in
            .sample(20)
            .join(web, how='inner')
    )
    topn['Recipe'] = "[" + topn['name'] + "](" + topn['url'] + ")"
    topn['similarity_score'] = 'Random'
    return topn

# Calculate similarity scores
def individual_similarity_scores(user_live, users_in, recipe2clust, clust_dist, recipes_in, ingr_vec):
    # start_time = time()
    ### Community Similarity ###
    live_user_recipes = users_in.loc[users_in['user_id'] == user_live] # Gets all recipes live user has rated
    live_user_clusters = live_user_recipes.merge(recipe2clust, on='recipe_id', how='inner') # Gets cluster assignments for these recipes

    unique_clusters = live_user_clusters[['community_id']].groupby('community_id', as_index=False).size()  # Unique clusters from recipes live user has rated
    all_clusters = recipe2clust['community_id'].unique() # All clusters
    combos = unique_clusters.copy()
    combos['community_id_2'] = pd.NA

    for i in range(len(combos)): combos.iat[i,2] = all_clusters

    live_dist = combos.explode('community_id_2').merge(clust_dist, on=['community_id', 'community_id_2'], how='left') # Gets all cluster pairs from live user and merges distances
    community_sim = live_dist.merge(recipe2clust, left_on='community_id_2', right_on='community_id', how='left')#.drop_duplicates() # Get recipe IDs for all clusters with distances
    community_sim['similarity'] = community_sim['similarity'] * community_sim['size']
    community_sim['similarity'].fillna(0, inplace=True) # NA values are where source node = target node
    np.random.seed(0)
    community_sim['similarity'] = community_sim['similarity'] + np.random.normal(0, 0.025, size=community_sim.shape[0])
    community_sim = community_sim[['recipe_id', 'size', 'similarity', 'community_id_2']].groupby(['recipe_id', 'community_id_2'], as_index=False).sum()
    community_sim['similarity'] = community_sim['similarity'] / community_sim['size']
    community_sim = community_sim.drop(['size', 'community_id_2'], axis=1)
    community_sim = community_sim.set_index('recipe_id')
    community_sim = (community_sim - community_sim.min())/(community_sim.max()-community_sim.min()) * 2 - 1

    ### Cuisine Similarity ###
    cuisines = recipes_in['cuisine'].unique() # All unique cuisines
    user_cuisines = pd.DataFrame(live_user_recipes.merge(recipes_in, on='recipe_id')['cuisine'].value_counts()) # Value counts of cuisines for recipes user has rated
    all_cuisines = pd.DataFrame(cuisines).set_index(0) # Creates dataframe with zero counts for all cuisines
    all_cuisines['cuisine'] = 0
    user_cuisines = pd.concat([user_cuisines,all_cuisines]) # Combines user cuisine counts and zero count dataframe
    user_cuisines = user_cuisines[~user_cuisines.index.duplicated(keep='first')] # Remove duplicate (zero) cuisines
    user_cuisines = (user_cuisines-user_cuisines.min())/(user_cuisines.max()-user_cuisines.min()) # Scale between 0 and 1
    cuisine_sim = ( # Merges cuisine similarity score with all recipes
        recipes_in
        .reset_index()
        .merge(user_cuisines,left_on='cuisine', right_on=user_cuisines.index)
        .set_index('recipe_id')[['cuisine_y']]
        .rename(columns={'cuisine_y': 'cuisine_sim'})
    )
    cuisine_sim = (cuisine_sim - cuisine_sim.min())/(cuisine_sim.max()-cuisine_sim.min())* 2 - 1

    ### Technique Similarity ###
    tech_mat = np.array(recipes_in['techniques'].apply(lambda x: list(x.values())).tolist()) # Technique matrix to np array
    user_techniq = np.sum(tech_mat[recipes_in.index.isin(live_user_recipes['recipe_id']),:], axis=0) # Live user recipe technique

    dotp = np.sum(tech_mat * user_techniq,axis=1) # Numerator of cosine similarity
    denom = np.linalg.norm(tech_mat,axis=1) * np.linalg.norm(user_techniq) # Denominator of cosine similarity
    a = np.divide(dotp,denom,out=np.zeros_like(denom), where=denom!=0) # Cosine similarity, places where denom is zero are set to zero

    techniq_sim = pd.DataFrame(a,index=recipes_in.index,columns=['technique_sim'])
    techniq_sim = (techniq_sim - techniq_sim.min())/(techniq_sim.max()-techniq_sim.min())* 2 - 1

    ### Ingredient Similarity ###
    ingr_mat = np.array(ingr_vec['recipe_ingredients_vector'].tolist()) # Ingredient matrix to np array
    user_ingr = np.sum(ingr_mat[ingr_vec.index.isin(live_user_recipes['recipe_id']),:],axis=0) # Live user recipe ingredients

    dotp = np.sum(ingr_mat * user_ingr,axis=1) # Numerator of cosine similarity
    denom = np.linalg.norm(ingr_mat,axis=1) * np.linalg.norm(user_ingr) # Denominator of cosine similarity
    a = np.divide(dotp,denom,out=np.zeros_like(denom), where=denom!=0) # Cosine similarity, places where denom is zero are set to zero

    ingr_sim = pd.DataFrame(a,index=ingr_vec.index,columns=['ingr_sim'])
    ingr_sim = (ingr_sim - ingr_sim.min())/(ingr_sim.max()-ingr_sim.min())* 2 - 1

    ### Aggregate Similarity ###
    agg_sim = community_sim.join(cuisine_sim,how='inner').join(techniq_sim,how='inner').join(ingr_sim,how='inner')
    # end_time = time()
    # print("updating similarity scores takes {} seconds.".format(end_time - start_time))
    return agg_sim

# Calculate aggregate similarity score
def aggregate_similarity(individual_similarity, weight_ingr, weight_techniq, weight_cuisine, weight_community):
    individual_similarity['similarity_score'] = individual_similarity['similarity'] * weight_community + \
        individual_similarity['cuisine_sim'] * weight_cuisine + \
        individual_similarity['technique_sim'] * weight_techniq + \
        individual_similarity['ingr_sim'] * weight_ingr
    return individual_similarity[['similarity_score']]


# Compute all similarities on startup with default user
user_similarities = individual_similarity_scores(788, users_in, recipe2clust, clust_dist, recipes_in, ingr_vec)
current_user_id = 788
prev_user_id = 788



@app.callback(Output('user_id_input', 'value'),
              Input('url', 'pathname'))
def global_user_id_value(pathname):
    global total_clicks
    total_clicks = 0
    return current_user_id


@app.callback(Output('dummy_2', 'children'),
              Input('user_id_input', 'value'))
def update_global_vars(selected_user):
    global current_user_id
    global prev_user_id
    prev_user_id = current_user_id
    current_user_id = selected_user
    return []


########### GRAPHS - USER SUMMARIES / NETWORK GRAPH FOR BOTH ############

# Ingredient graph - user summaries
@app.callback(
    dash.dependencies.Output('ing_graph', 'figure'),
    dash.dependencies.Input('user_id_input', 'value'))
def update_graph1(selected_user):

    # start_time = time()

    live_ingr_ct_filter = live_ingr_ct[live_ingr_ct['num_ingredients'] <= 200]
    filter_user_1 = np.array(live_ingr_ct[live_ingr_ct['user_id'] == selected_user])[0, 1]

    trace1 = go.Histogram(
        x=live_ingr_ct_filter.num_ingredients,
        opacity=0.75,
        histnorm='probability',
        name='Typical Ingredient Usage',
        nbinsx=36,
        marker=dict(color="#8d8d8d")
    )

    freqs, _ = np.histogram(live_ingr_ct_filter.num_ingredients, bins=36, density = True)
    max_freq = np.max(freqs)*10

    trace2 = go.Scatter(
                x=[filter_user_1,filter_user_1],
                y=[0,max_freq],
                mode='lines',
                line={
                    'color': '#489FB5',
                    'width': 2,
                    'dash': 'solid',
                }, name='Your Ingredient Usage'
    )

    data = [trace1, trace2]

    layout = go.Layout(
        title='Ingredients',
        barmode='overlay',
        xaxis=dict(
            title='Number of Ingredients'
        ),
        yaxis=dict(
            title='Frequency',
        ),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        showlegend=True
    )

    hist_fig_1 = go.Figure(data=data, layout=layout)
    hist_fig_1.update_yaxes(showline=True, linewidth=1, linecolor='#EDEBE9', gridcolor='#EDEBE9')
    hist_fig_1['layout']['yaxis']['showgrid'] = False
    # end_time = time()
    # print("updating ingredient graph takes {} seconds.".format(end_time - start_time))
    return hist_fig_1

# Cuisine graph - user summaries
@app.callback(
    dash.dependencies.Output('cui_graph', 'figure'),
    dash.dependencies.Input('user_id_input', 'value'))
def update_graph2(selected_user):
    # start_time = time()

    user_cuis = live_cusi[live_cusi['user_id'] == selected_user]
    user_cuis['pct'] = user_cuis['count'] / sum(user_cuis['count'])
    all_live_cuis = live_cusi[['cuisine', 'count']].groupby('cuisine', as_index=False).sum()
    all_live_cuis['pct'] = all_live_cuis['count'] / sum(all_live_cuis['count'])
    all_live_cuis = all_live_cuis.sort_values(by='pct', ascending=False)

    cuis_layout = dict(
        title=f'Cuisines',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(  # linecolor='black',
            showgrid=True,
            showticklabels=True,
            mirror=True),
        yaxis=dict(  # linecolor='black',
            showgrid=True,
            showticklabels=True,
            mirror=True),
        yaxis_title='Cuisine',
        xaxis_title='Proportion of Usage')

    cuis_fig = go.Figure(
        data=[go.Bar(x=all_live_cuis["cuisine"], y=all_live_cuis["pct"], name="Typical Cuisines",
                     marker={'color': '#8d8d8d'}, hovertemplate='<b>%{x}</b>'+'<br>Proportion: %{y:.2f}'),
              go.Bar(x=user_cuis["cuisine"], y=user_cuis["pct"], name="Your Cuisines",
                     marker={'color': '#489FB5'}, hovertemplate='<b>%{x}</b>'+'<br>Proportion: %{y:.2f}')],
        layout=cuis_layout)
    cuis_fig.update_layout(barmode="group")
    # end_time = time()
    # print("updating cuisine graph takes {} seconds.".format(end_time - start_time))
    return cuis_fig

# Technique graph - user summaries
@app.callback(
    dash.dependencies.Output('tech_graph', 'figure'),
    dash.dependencies.Input('user_id_input', 'value'))
def update_graph3(selected_user):
    # start_time = time()
    user_tech = live_tech[live_tech['user_id'] == selected_user]
    user_tech['pct'] = user_tech['count'] / sum(user_tech['count'])
    user_tech = user_tech.sort_values(by='count')
    all_live_tech = live_tech[['techniques', 'count']].groupby('techniques', as_index=False).sum()
    all_live_tech['pct'] = all_live_tech['count'] / sum(all_live_tech['count'])

    tech_data = all_live_tech.merge(user_tech, on='techniques', how='left', suffixes=('_all', '_user')).fillna(0)
    tech_data = tech_data[(tech_data.index <= 15) | (tech_data['pct_user'] > 0)]
    tech_data = tech_data.sort_values(by='pct_all')

    tech_layout = dict(
        title=f'Techniques',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(  # linecolor='black',
            showgrid=True,
            showticklabels=True,
            mirror=True),
        yaxis=dict(  # linecolor='black',
            showgrid=True,
            showticklabels=True,
            mirror=True),
        yaxis_title='Technique',
        xaxis_title='Proportion of Usage',
        height=600,
        margin=dict(t=50, l=125))

    tech_fig = go.Figure(
        data=[go.Bar(y=tech_data["techniques"], x=tech_data["pct_all"], name="Typical Techniques",
                     marker={'color': '#8d8d8d'}, hovertemplate='<b>%{y}</b>'+'<br>Proportion: %{c:.2f}',
                     orientation='h'),
              go.Bar(y=tech_data["techniques"], x=tech_data["pct_user"], name="Your Techniques",
                     marker={'color': '#489FB5'}, hovertemplate='<b>%{y}</b>'+'<br>Proportion: %{x:.2f}',
                     orientation='h')],
        layout=tech_layout)
    tech_fig.update_layout(barmode="group")
    # end_time = time()
    # print("updating technique graph takes {} seconds.".format(end_time - start_time))
    return tech_fig

# Wordcloud graph - user summaries
@app.callback(
    dash.dependencies.Output('wordcloud_graph', 'figure'),
    dash.dependencies.Input('user_id_input', 'value'))
def update_graph4(selected_user):
    # start_time = time()
    filter_user_1 = live_ingr[live_ingr['user_id'] == selected_user]
    d = dict(zip(filter_user_1['ingredients'], filter_user_1['count']))
    wordcloud = WordCloud(font_path='assets/Helvetica.ttc', height=400, background_color='#ffffff',color_func=lambda *args,
        **kwargs: (82, 82, 82),collocation_threshold=0, min_font_size=20).generate_from_frequencies(d)
    wordmap = px.imshow(wordcloud, title=f'Your Most Used Ingredients')
    wordmap.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    wordmap.update_layout(coloraxis_showscale=False)
    # end_time = time()
    # print("updating wordcloud takes {} seconds.".format(end_time - start_time))
    return wordmap


def networkGraph(selected_user):
    edges_adj = edges[edges['community_id'] != edges['community_id_2']]
    edges_adj = edges_adj[edges_adj['pct_users_mean'] > .135]

    user_counts = pd.read_parquet('data/user_assign_coummunity_with_recipe_count.parquet')
    user_counts = user_counts[user_counts['user_id'] == selected_user].drop('user_id', axis=1)

    node_sizes = pd.read_csv('data/node_sizes.csv', delimiter=',')
    node_sizes['count'] = node_sizes['count'] / 10
    node_sizes = dict(np.array(node_sizes[['community_id', 'count']]))

    node_colors = pd.DataFrame(np.array([list(node_sizes.keys()), np.zeros(len(node_sizes))]).T,
                               columns=['community_id', 'user_count'])
    node_colors = node_colors.merge(user_counts, on='community_id', how='left')
    node_colors['count'][node_colors['count'].isna()] = 0
    node_colors = dict(np.array(node_colors[['community_id', 'count']]))

    G = nx.from_pandas_edgelist(df=edges_adj, source='community_id', target='community_id_2',
                                edge_attr='pct_users_mean')

    pos = nx.spring_layout(G, weight='pct_users_mean', k=0.9)

    # edges trace
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(color='lightgray', width=0.5),
        hoverinfo='none',
        showlegend=False,
        mode='lines')

    # nodes trace
    node_x = []
    node_y = []
    text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(node)

    node_sizes_filt = [node_sizes[node] / 8 for node in G.nodes()]
    node_colors_filt = [node_colors[node] for node in G.nodes()]
    node_edge_colors_filt = ['black' if node_colors[node] > 0 else 'gray' for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        # text= node_colors['community_id'],

        mode='markers',
        showlegend=False,
        hoverinfo='none',
        marker=dict(
            colorscale='Blues',
            color=node_colors_filt,
            size=node_sizes_filt,
            line=dict(color=node_edge_colors_filt, width=1)))

    # layout
    layout = dict(
        title=f'User Community Network',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(  # linecolor='black',
            showgrid=False,
            showticklabels=False,
            mirror=True),
        yaxis=dict(  # linecolor='black',
            showgrid=False,
            showticklabels=False,
            mirror=True))

    # figure
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)

    return fig

# Create network graph for Try New Recipes page
def networkGraphRecs(selected_user, rec_recipes):
    edges = pd.read_parquet('data/community_graph_edges_pctthresh-08_split.parquet')
    edges_adj = edges[edges['community_id'] != edges['community_id_2']]
    edges_adj = edges_adj[edges_adj['pct_users_mean'] > .135]

    user_counts = pd.read_parquet('data/user_assign_coummunity_with_recipe_count.parquet')
    user_counts = user_counts[user_counts['user_id'] == selected_user].drop('user_id', axis=1)

    node_sizes = pd.read_csv('data/node_sizes.csv', delimiter=',')
    node_sizes['count'] = node_sizes['count'] / 15
    node_sizes = dict(np.array(node_sizes[['community_id', 'count']]))

    rec_comms = rec_recipes.merge(recipe2clust, on='recipe_id')
    rec_comms = rec_comms[['recipe_id', 'community_id']].groupby('community_id', as_index=False).count() \
                                                            .rename({'recipe_id':'count'}, axis=1)
    edge_thickness = pd.DataFrame(np.array([list(node_sizes.keys()), np.zeros(len(node_sizes))]).T,
                 columns=['community_id', 'user_count'])
    edge_thickness = edge_thickness.merge(rec_comms, on='community_id', how='left')
    edge_thickness['count'][edge_thickness['count'].isna()] = 0
    edge_thickness = dict(np.array(edge_thickness[['community_id', 'count']]))

    node_colors = pd.DataFrame(np.array([list(node_sizes.keys()), np.zeros(len(node_sizes))]).T,
                               columns=['community_id', 'user_count'])
    node_colors = node_colors.merge(user_counts, on='community_id', how='left')
    node_colors['count'][node_colors['count'].isna()] = 0
    node_colors = dict(np.array(node_colors[['community_id', 'count']]))

    G = nx.from_pandas_edgelist(df=edges_adj, source='community_id', target='community_id_2',
                                edge_attr='pct_users_mean')

    pos = nx.spring_layout(G, weight='pct_users_mean', k=0.7)

    # edges trace
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(color='lightgray', width=0.5),
        hoverinfo='none',
        showlegend=False,
        mode='lines')

    # nodes trace
    node_x = []
    node_y = []
    text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(node)

    node_sizes_filt = [node_sizes[node] / 8 for node in G.nodes()]
    node_colors_filt = [node_colors[node] for node in G.nodes()]
    node_edge_colors_filt = ['#FFA62B' if edge_thickness[node] > 0 else 'gray' for node in G.nodes()]
    node_edge_thickness_filt = [edge_thickness[node] for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        # text= node_colors['community_id'],

        mode='markers',
        showlegend=False,
        hoverinfo='none',
        marker=dict(
            colorscale='Blues',
            color=node_colors_filt,
            size=node_sizes_filt,
            line=dict(color=node_edge_colors_filt, width=node_edge_thickness_filt)))

    # layout
    layout = dict(
        title=f'Community Network with Recommendations',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(  # linecolor='black',
            showgrid=False,
            showticklabels=False,
            mirror=True),
        yaxis=dict(  # linecolor='black',
            showgrid=False,
            showticklabels=False,
            mirror=True))

    # figure
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)

    return fig

# Update network graph on user summaries page
@app.callback(
    dash.dependencies.Output('network_graph', 'figure'),
    dash.dependencies.Input('user_id_input', 'value'))
def update_output(selected_user):
    # start_time = time()
    new_graph = networkGraph(selected_user)
    # end_time = time()
    # print("updating network graph takes {} seconds.".format(end_time - start_time))

    return new_graph


# Create dependency between user similarity calc and recs to avoid lag issues
@app.callback(
    Output('dummy_1', 'children'),
    Input('dummy_2', 'children')
)
def update_sim_scores(_):
    if current_user_id != prev_user_id:
        global user_similarities
        user_similarities = individual_similarity_scores(current_user_id, users_in, recipe2clust, clust_dist, recipes_in,
                                                     ingr_vec)  # Compute all similarities on startup
    return []

########################################
####### TRY NEW RECIPES ################
########################################
@app.callback(
    Output('rec-table', 'data'),
    Output('wordcloud_graph_user', 'figure'),
    Output('wordcloud_graph_recs', 'figure'),
    Output('technique_graph_recs', 'figure'),
    Output('cuisine_graph_recs', 'figure'),
    Output('network_graph_recs', 'figure'),
    Output('recipe_rec_row', 'style'),
    Output('rec_vis_row', 'style'),
    Output('top-recipes-recs-title','children'),
    Input('gen-recipes', 'n_clicks'),
    Input('rec-table', 'data'),
    Input('calories-dropdown', 'value'),
    Input('carbs-dropdown', 'value'),
    Input('sugar-dropdown', 'value'),
    Input('satfat-dropdown', 'value'),
    Input('protein-dropdown', 'value'),
    Input('minute-dropdown', 'value'),
    Input('meal-dropdown','value'),
    Input('community-slider','value'),
    Input('techniq_slider','value'),
    Input('ingr-slider','value'),
    Input('cuisine-slider','value'),
    dash.dependencies.Input('user_id_input', 'value'),
    Input('recipe_rec_row', 'style'),
    Input('rec_vis_row', 'style'),
    Input('dummy_1', 'children')
)
def generate_recipe_recommendations(button_clicks, old_table, calories, carbs, sugar, satfat, protein, minute, meal,
                                    community, techniq, ingr, cuisine, selected_user, recipe_table_style, rec_vis_style,
                                    _):
    global total_clicks

    if button_clicks and button_clicks > total_clicks:
        # start_time = time()
        # Nutrition Filter
        nutrition_filter = []
        if calories: nutrition_filter.append(nutrition['calories'] <= calories)
        if carbs: nutrition_filter.append(nutrition['carbohydrates']<= carbs)
        if protein: nutrition_filter.append(nutrition['protein']<= protein)
        if satfat: nutrition_filter.append(nutrition['saturated fat']<= satfat)
        if sugar: nutrition_filter.append(nutrition['sugar']<= sugar)
        # Other filter
        other_filter = []
        if minute: other_filter.append(recipes_in['minutes']<=minute)
        if type(meal) is str: meal = [meal]
        if meal: other_filter.append(recipes_in['meal_of_day'].isin(meal))
        # Apply filters
        if nutrition_filter:
            new_nutrition = nutrition.loc[np.logical_and.reduce(nutrition_filter),:]
        else: new_nutrition=pd.DataFrame()

        if other_filter:
            new_recipes_in = recipes_in.loc[np.logical_and.reduce(other_filter)]
        else: new_recipes_in = recipes_in.copy()

        if len(new_nutrition) > 0:
            new_recipes_in = new_recipes_in.loc[new_recipes_in.index.isin(new_nutrition['recipe_id']),:]
        # Update Recommendations
        total_weight = sum(np.abs([ingr,techniq,cuisine,community]))
        if total_weight != 0:
            ingr = ingr/total_weight
            techniq =techniq/total_weight
            cuisine = cuisine/total_weight
            community = community/total_weight
            new_user_similarities = user_similarities.loc[user_similarities.index.isin(new_recipes_in.index),:]
            updated_sim = aggregate_similarity(new_user_similarities,ingr,techniq,cuisine,community)

            new_recipes_topn = create_recipe_table(new_recipes_in, website, updated_sim)
            updated_table = new_recipes_topn[['Recipe']]

            recipe_rec_title = 'TOP RECIPE RECOMMENDATIONS'

        else:
            new_recipes_topn = create_random_recipe_table(new_recipes_in, website)
            updated_table = new_recipes_topn[['Recipe']]

            recipe_rec_title = 'RANDOM RECIPES'


        total_clicks = button_clicks

        # end_time = time()
        # print("generating new recipes takes {} seconds.".format(end_time - start_time))

        # Make recommendation wordcloud
        # start_time = time()
        new_recipes_in_ings = new_recipes_topn.reset_index()[['ingredients', 'recipe_id']].explode('ingredients')
        new_recipes_in_ings = new_recipes_in_ings.groupby('ingredients', as_index=False).count() \
                                                    .rename({'recipe_id':'count'}, axis=1) \
                                                    .sort_values('count', ascending=False).iloc[:15, :]
        d = dict(zip(new_recipes_in_ings['ingredients'], new_recipes_in_ings['count']))
        wordcloud = WordCloud(font_path='assets/Helvetica.ttc', height=400, background_color='#ffffff',
                              color_func=lambda *args, **kwargs: (82, 82, 82),
                              collocation_threshold=0, min_font_size=20).generate_from_frequencies(d)

        rec_wordmap = px.imshow(wordcloud, title=f'Recommended Ingredients')
        rec_wordmap.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        rec_wordmap.update_layout(coloraxis_showscale=False)

        filter_user_1 = live_ingr[live_ingr['user_id'] == selected_user]
        d = dict(zip(filter_user_1['ingredients'], filter_user_1['count']))
        wordcloud = WordCloud(font_path='assets/Helvetica.ttc', height=400, background_color='#ffffff',
                              color_func=lambda *args, **kwargs: (82, 82, 82),
                              collocation_threshold=0, min_font_size=20).generate_from_frequencies(d)

        wordmap = px.imshow(wordcloud, title=f'Your Most Used Ingredients')
        wordmap.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        wordmap.update_layout(coloraxis_showscale=False)

        # end_time = time()
        # print("updating wordclouds (recs) takes {} seconds.".format(end_time - start_time))

        # Make recommendation techniques graph
        # start_time = time()
        user_techs = live_tech[live_tech['user_id'] == selected_user]
        user_techs['pct'] = user_techs['count'] / sum(user_techs['count'])
        tech_recs = new_recipes_topn.reset_index()[['recipe_id', 'techniques']]
        tech_recs['techniques'] = tech_recs['techniques'].apply(lambda x: [k for k, v in x.items() if v > 0])
        tech_recs = tech_recs.explode('techniques')
        tech_recs = tech_recs.groupby('techniques', as_index=False).count().rename({'recipe_id': 'count'}, axis=1)
        tech_recs['techniques'] = tech_recs['techniques'].apply(lambda x: x.title())
        tech_recs['pct'] = tech_recs['count'] / sum(tech_recs['count'])

        tech_data = tech_recs.merge(user_techs, on='techniques', how='outer', suffixes=('_all', '_user')).fillna(0)
        tech_data = tech_data.sort_values(by='pct_all')

        tech_layout = dict(
            title=f'Techniques',
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(  # linecolor='black',
                showgrid=True,
                showticklabels=True,
                mirror=True),
            yaxis=dict(  # linecolor='black',
                showgrid=True,
                showticklabels=True,
                mirror=True),
            yaxis_title='Technique',
            xaxis_title='Proportion of Usage',
            height=600)

        tech_recs_fig = go.Figure(
            data=[go.Bar(y=tech_data["techniques"], x=tech_data["pct_all"], name="Recommended Techniques",
                         marker={'color': '#FFA62B'}, hovertemplate='<b>%{y}</b>' + '<br>Proportion: %{c:.2f}',
                         orientation='h'),
                  go.Bar(y=tech_data["techniques"], x=tech_data["pct_user"], name="Your Techniques",
                         marker={'color': '#489FB5'}, hovertemplate='<b>%{y}</b>' + '<br>Proportion: %{x:.2f}',
                         orientation='h')],
            layout=tech_layout)
        tech_recs_fig.update_layout(barmode="group")
        # end_time = time()
        # print("updating techniques graph (recs) takes {} seconds.".format(end_time - start_time))

        # Make recommendation cuisine graph

        # start_time = time()
        cuis_layout = dict(
            title=f'Cuisines',
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(  # linecolor='black',
                showgrid=True,
                showticklabels=True,
                mirror=True),
            yaxis=dict(  # linecolor='black',
                showgrid=True,
                showticklabels=True,
                mirror=True
            ),
            yaxis_title = 'Cuisine',
            xaxis_title = 'Proportion of Usage',
            margin=dict(t=50, l=125))

        user_cuis = live_cusi[live_cusi['user_id'] == selected_user]
        user_cuis['pct'] = user_cuis['count'] / sum(user_cuis['count'])
        rec_cuis = new_recipes_topn.reset_index()[['recipe_id', 'cuisine']]
        rec_cuis['cuisine'] = rec_cuis['cuisine'].apply(lambda x: x.title())
        rec_cuis = rec_cuis.groupby('cuisine', as_index=False).count().rename({'recipe_id': 'count'}, axis=1)
        rec_cuis['pct'] = rec_cuis['count'] / sum(rec_cuis['count'])
        cuis_recs_fig = go.Figure(
            data=[go.Bar(x=rec_cuis["cuisine"], y=rec_cuis["pct"], name="Recommended Cuisines",
                         marker={'color': '#FFA62B'}),
                  go.Bar(x=user_cuis["cuisine"], y=user_cuis["pct"], name="Your Cuisines",
                         marker={'color': '#489FB5'})],
            layout=cuis_layout)
        cuis_recs_fig.update_layout(barmode="group")
        # end_time = time()
        # print("updating cuisine graph (recs) takes {} seconds.".format(end_time - start_time))

        # Make recommendation network graph
        # start_time = time()
        networkgraph_recs = networkGraphRecs(selected_user, new_recipes_topn)
        # end_time = time()
        # print("updating network graph (recs) takes {} seconds.".format(end_time - start_time))

        recipe_table_style['display'] = 'flex'
        rec_vis_style['display'] = 'flex'

        return updated_table.to_dict('records'), wordmap, rec_wordmap, tech_recs_fig, cuis_recs_fig, networkgraph_recs, recipe_table_style, rec_vis_style, recipe_rec_title
    else:
        recipe_table_style['display'] = 'none'
        rec_vis_style['display'] = 'none'
        return old_table, {}, {}, {}, {}, {}, recipe_table_style, rec_vis_style, {}
