#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Plotly
import plotly.express as px
import plotly.graph_objects as go

# Dash
import dash
#from jupyter_dash import JupyterDash
import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table

# Data processing
import math
import json
import pandas as pd
import geopandas as gpd
import numpy as np

# Other
import os


# In[2]:


path = os.getcwd()
housing = pd.read_csv(os.path.join(path, "housing_final.csv"))
licenses = pd.read_csv(os.path.join(path,"licenses_final.csv"))
trading_mapping = {"Bottle shop": 0,
                  "Wholesaler": 0,
                  "Trading to 11PM": 1,
                  "Trading to Midnight": 2,
                  "Trading to 1AM": 3,
                  "Trading to 2AM": 4,
                  "Trading to 3AM": 5,
                  "Trading to 4AM": 6,
                  "Trading to 5AM": 7,
                  "Trading to 6AM": 8,
                  "Trading to 7AM": 9,
                  "Trading to 8AM": 10,
                  "Trading 24 Hours": 11}
licenses['Trading mapping'] = [trading_mapping[i] for i in licenses['After 11 pm']]


# In[3]:


df_right = licenses.groupby("geom_suburb").count()['Suburb'].reset_index()

df_merge = pd.merge(left = housing, right = df_right, left_on = 'geom_suburb', right_on = 'geom_suburb', how = 'left')
df_merge = df_merge.rename({"Suburb_y": "Bars"}, axis=1)
df_merge['Properties per Bar'] = df_merge['Propertycount'] / df_merge['Bars']


# In[4]:


shape_path = r"AUS Shape files\AUS_adm2.shp"
shapes = gpd.read_file(os.path.join(path, shape_path))
shapes = shapes[shapes.NAME_1 == "Victoria"].reset_index(drop=True)
shapes.NAME_2 = shapes.NAME_2.str.replace("S'bank-D'lands", "Southbank - Docklands")


# # Styling

# In[5]:


# Style dicts
app_style = {
    "backgroundColor": "#292F35",
    "color": "#FFFDFA",
    "font-family": "Helvetica",
    "padding-left": 25,
    "padding-right": 25,
    "padding-top": 15,
    "padding-bottom": 15,
}

subdiv_style = {
    "backgroundColor": "#292F35",
    "color": "#FFFDFA",
    "font-family": "Helvetica",
    "padding-left": 0,
    "padding-right": 0,
    "padding-top": 15,
    "padding-bottom": 15
}

kpi_style = {
    "backgroundColor": "#292F35",
    "color": "#FFFDFA",
    "font-family": "Helvetica",
    "padding-left": 0,
    "padding-right": 0,
    "padding-top": 15,
    "padding-bottom": 15,
    "textAlign" : "center"
}

grid_style = {
    "backgroundColor": "#292F35",
    "color": "#FFFDFA",
    "font-family": "Helvetica",
    "font-size": 11,
    "padding-left": 0,
    "padding-right": 0,
    "padding-top": 15,
    "padding-bottom": 15,
    "textAlign" : "center"
}

link_style = {
    "font-weight": "bold",
    "font-family": "Helvetica",
    "font-size": "16px",
    "textAlign": "center",
    "color": "#ffffff",
    #'text-decoration': 'none',
    'text-decoration-color': '#ffffff'
}


# # Filter dictionaries

# In[6]:


# General Statistics
suburb = sorted(housing.geom_suburb.unique().tolist())
suburb_class = [{'label': str(item),
                      'value': str(item)}
                     for item in suburb]


# In[7]:


# Housing
housing_type = ["House", "Townhouse", "Unit"]
housing_value = ["h", "t", "u"]
housing_class = [{"label": str(housing_type[i]), "value": str(housing_value[i])} for i in range(len(housing_type))]

rooms_class = {item: str(item) for item in range(housing.Rooms.min(), housing.Rooms.max()+1)}

distance_class = {item: str(item) for item in range(0, (int(math.ceil(housing.Distance.max() / 10.0)) * 10)+1, 5)}


# In[8]:


# Area niceness
trading_hours = {"Bottle shops & Wholesalers": 0,
                  "Trading to 11PM": 1,
                  "Trading to Midnight": 2,
                  "Trading to 1AM": 3,
                  "Trading to 2AM": 4,
                  "Trading to 3AM": 5,
                  "Trading to 4AM": 6,
                  "Trading to 5AM": 7,
                  "Trading to 6AM": 8,
                  "Trading to 7AM": 9,
                  "Trading to 8AM": 10,
                  "Trading 24 Hours": 11}
trading_class = [{"label": str(key), "value": str(val)} for key, val in trading_hours.items()]

category = sorted(licenses['Category'].unique().tolist())
category_class = [{"label": str(i), "value": i} for i in category]


# In[9]:


# Model page
bathroom_class = {item: str(item) for item in range(int(df_merge.Bathroom.min()), int(df_merge.Bathroom.max()+1), 2)}
car_class = {int(item): str(int(item)) for item in range(int(df_merge.Car.min()), int(df_merge.Car.max())+1, 2)}


# # General Statistics

# In[10]:


general_stats_layout = html.Div(style=app_style,children=[
    # Title - Row
    html.Div(
        [
            html.H1(
                'General Statistics',
                className='five columns',
            ),
            html.Div([ 
                dcc.Link('Home', href='/', style = link_style)
                    ], className = 'one columns', style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Div([
                dcc.Link('General Statistics', href='/general_statistics', style=link_style) 
            ], className = "one columns", style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Div([
                dcc.Link('Housing overview', href='/housing_overview', style = link_style) 
            ], className = "one columns", style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Div([
                dcc.Link('Area niceness', href='/area_niceness', style = link_style) 
            ], className = "one columns", style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Div([
                dcc.Link('Price prediction', href='/model_page', style = link_style) 
            ], className = "one columns", style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Img(
                src="http://logok.org/wp-content/uploads/2014/06/City-of-Melbourne-logo-M-880x660.png",
                className='two columns',
                style={
                    'height': '12%',
                    'width': '12%',
                    'float': 'right',
                    'position': 'top',
                    'padding-top': 0,
                    'padding-right': 0
                },
            ),
            html.P(
                'Select your preferred suburb and see which features has the biggest influence on the price of the different property types - Or scroll through a few areas to see the difference in the parameter importance',
                className='eight columns',
            ),
        ],
        className='row'
    ),
    # Selectors
    html.Div(
        [
            html.Div(
                [                    
                    html.P('Suburb:'),
                    dcc.Dropdown(
                        id='suburb_drop_stats',
                        options= suburb_class,
                        multi=False,
                        style={"color": "#616A6B"}
                    )
                ],
                className='three columns',
                style={'margin-top': '10'}
            ),           
        ], style=subdiv_style, className='row'
    ),
    # Create KPI divs
    html.Div(children=
        [
            html.Div(style={"backgroundColor": "#5D6D7E", "textAlign": "center"}, children=
                [
                    html.Div(id = "avg_price_stats"),
                    html.P("Average price (AUD)")
                ], className = "three columns"
            ),
            html.Div(style={"backgroundColor": "#5D6D7E", "textAlign": "center"}, children=
                [
                html.Div(id = "no_listings_stats"),
                html.P("Housing for sale")
                ], className = "three columns"
            ),
            html.Div(style={"backgroundColor": "#5D6D7E", "textAlign": "center"}, children=
                [
                    html.Div(id = "no_bars_stats"),
                    html.P("Avg. # of bars in area")
                ], className = "three columns"
            ),
            html.Div(style={"backgroundColor": "#5D6D7E", "textAlign": "center"}, children=
                [
                    html.Div(id = "no_properties_stats"),
                    html.P("Total # of properties in area")
                ], className = "three columns"
            ),
                
        ], style=kpi_style, className = "row"
    ),
    # Create house row
    html.Div(
        [
            html.Div([
                html.P("Distance to CBD"),
                dcc.Graph(id = "grid1_1"),
                dcc.Graph(id = "grid2_1"),
                dcc.Graph(id = "grid3_1")], 
                className = "two columns"),
            html.Div([
                html.P("Rooms"),
                dcc.Graph(id = "grid1_2"),
                dcc.Graph(id = "grid2_2"),
                dcc.Graph(id = "grid3_2")],
                className = "two columns"),
            html.Div([
                html.P("Bathroom"),
                dcc.Graph(id = "grid1_3"),
                dcc.Graph(id = "grid2_3"),
                dcc.Graph(id = "grid3_3")], 
                className = "two columns"),
            html.Div([
                html.P("Car"),
                dcc.Graph(id = "grid1_4"),
                dcc.Graph(id = "grid2_4"),
                dcc.Graph(id = "grid3_4")], 
                className = "two columns"),
            html.Div([
                html.P("Properties per bar"),
                dcc.Graph(id = "grid1_5"),
                dcc.Graph(id = "grid2_5"),
                dcc.Graph(id = "grid3_5")], 
                className = "two columns"),
            html.Div([
                html.P("Suburb Area"),
                dcc.Graph(id = "grid1_6"),
                dcc.Graph(id = "grid2_6"),
                dcc.Graph(id = "grid3_6")], 
                className = "two columns"),
        ], style=grid_style, className="row"
    )
],
className = 'ten columns offset-by-one'
)


# # Housing

# In[11]:


housing_layout = html.Div(style=app_style,children=[
    # Title - Row
    html.Div(
        [
            html.H1(
                'Housing Overview',
                className='five columns',
            ),
            html.Div([ 
                dcc.Link('Home', href='/', style = link_style)
                    ], className = 'one columns', style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Div([
                dcc.Link('General Statistics', href='/general_statistics', style=link_style) 
            ], className = "one columns", style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Div([
                dcc.Link('Housing overview', href='/housing_overview', style = link_style) 
            ], className = "one columns", style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Div([
                dcc.Link('Area niceness', href='/area_niceness', style = link_style) 
            ], className = "one columns", style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Div([
                dcc.Link('Price prediction', href='/model_page', style = link_style) 
            ], className = "one columns", style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Img(
                src="http://logok.org/wp-content/uploads/2014/06/City-of-Melbourne-logo-M-880x660.png",
                className='two columns',
                style={
                    'height': '12%',
                    'width': '12%',
                    'float': 'right',
                    'position': 'top',
                    'padding-top': 0,
                    'padding-right': 0
                },
            ),
            html.P('Here you can get an overview of the general price differences between the different suburbs. Furthermore, you can see the price difference for a house, townhouse and a unit in the suburbs, and the general price trend depending on how far the address is from Melbourne CBD',
                className='eight columns',
            ),
        ],
        className='row'
    ),
    # Selectors
    html.Div(
        [
            html.Div(
                [                    
                    html.P('Suburb:'),
                    dcc.Dropdown(
                        id='suburb_drop_housing',
                        options= suburb_class,
                        multi=False,
                        style={"color": "#616A6B"}
                    )
                ],
                className='three columns',
                style={'margin-top': '10'}
            ),
            html.Div(
                [                    
                    html.P('Housing Type:'),
                    dcc.Dropdown(
                        id='type_drop_housing',
                        options= housing_class,
                        multi=False,
                        style={"color": "#616A6B"}
                    )
                ],
                className='three columns',
                style={'margin-top': '10'}
            ),
            html.Div(
                [                    
                    html.P('Rooms:'),
                    dcc.RangeSlider(
                        id='rooms_slider_housing',
                        min = housing.Rooms.min(),
                        max = housing.Rooms.max(),
                        marks = rooms_class,
                        value=[housing.Rooms.min(), housing.Rooms.max()],
                    )
                ],
                className='three columns',
                style={'margin-top': '10'}
            ),
            html.Div(
                [                    
                    html.P('Distance to CBD:'),
                    dcc.RangeSlider(
                        id='distance_slider_housing',
                        min = 0,
                        max = int(math.ceil(housing.Distance.max() / 10.0))*10,
                        marks = distance_class,
                        value=[0, int(math.ceil(housing.Distance.max() / 10.0))*10],
                    )
                ],
                className='three columns',
                style={'margin-top': '10'}
            )
        ], style=subdiv_style, className='row'
    ),
    # Create KPIs
    html.Div(children=
        [
            html.Div(style={"backgroundColor": "#5D6D7E", "textAlign": "center"}, children=
                [
                    html.Div(id = "avg_price_housing"),
                    html.P("Avg. price (AUD)")
                ], className = "three columns"
            ),
            html.Div(style={"backgroundColor": "#5D6D7E", "textAlign": "center"}, children=
                [
                html.Div(id = "no_listings_housing"),
                html.P("Houses for sale")
                ], className = "three columns"
            ),
            html.Div(style={"backgroundColor": "#5D6D7E", "textAlign": "center"}, children=
                [
                    html.Div(id = "no_bars_housing"),
                    html.P("Avg. # of bars in area")
                ], className = "three columns"
            ),
            html.Div(style={"backgroundColor": "#5D6D7E", "textAlign": "center"}, children=
                [
                    html.Div(id = "no_distance_housing"),
                    html.P("Avg. distance to CBD")
                ], className = "three columns"
            ),
                
        ], style=kpi_style, className = "row"
    ),
    # Create choropleth and plot next to it
    html.Div(children=
             [
                 html.Div([
                     dcc.Graph(id="choropleth_map_housing")
                 ], className="seven columns"
                 ),
                 html.Div([
                     dcc.Graph(id = "distance_hbar")
                 ], className = "five columns"
                 )
             ], style=kpi_style, className="row"
    ),
    # Create bottom plot
    html.Div(children=
             [
                 html.Div([
                     dcc.Graph(id="housing_price_bar")
                 ], className="twelve columns"
                 )
             ], style=kpi_style, className="row")
], className = "row"
)


# # Area niceness

# In[12]:


area_niceness_layout = html.Div(style=app_style,children=[
    # Title - Row
    html.Div(
        [
            html.H1(
                'Area niceness',
                className='five columns',
            ),
            html.Div([ 
                dcc.Link('Home', href='/', style = link_style)
                    ], className = 'one columns', style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Div([
                dcc.Link('General Statistics', href='/general_statistics', style=link_style) 
            ], className = "one columns", style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Div([
                dcc.Link('Housing overview', href='/housing_overview', style = link_style) 
            ], className = "one columns", style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Div([
                dcc.Link('Area niceness', href='/area_niceness', style = link_style) 
            ], className = "one columns", style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Div([
                dcc.Link('Price prediction', href='/model_page', style = link_style) 
            ], className = "one columns", style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Img(
                src="http://logok.org/wp-content/uploads/2014/06/City-of-Melbourne-logo-M-880x660.png",
                className='two columns',
                style={
                    'height': '12%',
                    'width': '12%',
                    'float': 'right',
                    'position': 'top',
                    'padding-top': 0,
                    'padding-right': 0
                },
            ),
            html.P(
                'Check out the restaurants, cafés, bars, clubs and bottle shops in your favorite area. The filters on license category and trading hours are closely connected, so it might only make sense to filter one at a time. Restaurants are considered as establishments open between 11PM and 1AM and night clubs are establishments open after 2AM.',
                className='eight columns',
            ),
        ],
        className='row'
    ),
    # Selectors
    html.Div(
        [
            html.Div(
                [                    
                    html.P('Suburb:'),
                    dcc.Dropdown(
                        id='suburb_drop_area',
                        options= suburb_class,
                        multi=False,
                        style={"color": "#616A6B"}
                    )
                ],
                className='three columns',
                style={'margin-top': '10'}
            ),
            html.Div(
                [                    
                    html.P('License category:'),
                    dcc.Dropdown(
                        id='category_drop_area',
                        options= category_class,
                        multi=False,
                        style={"color": "#616A6B"}
                    )
                ],
                className='three columns',
                style={'margin-top': '10'}
            ),
            html.Div(
                    [                    
                    html.P('Trading Hours after:'),
                    dcc.Dropdown(
                        id = "trading_drop_area", 
                        options=trading_class,
                        multi=False,
                        style={"color": "#616A6B"}
                    )  
                    ], className='three columns', style={'margin-top': '10'}
            )
        ], style=subdiv_style, className='row'
    ),
    # Create KPIs
    html.Div(children=
        [
            html.Div(style={"backgroundColor": "#5D6D7E", "textAlign": "center"}, children=
                [
                    html.Div(id = "no_bars_area"),
                    html.P("# of bars in area")
                ], className = "three columns"
            ),
            html.Div(style={"backgroundColor": "#5D6D7E", "textAlign": "center"}, children=
                [
                html.Div(id = "no_propbar_area"),
                html.P("Properties per Bar")
                ], className = "three columns"
            ),
            html.Div(style={"backgroundColor": "#5D6D7E", "textAlign": "center"}, children=
                [
                    html.Div(id = "no_nightclub_area"),
                    html.P("# of night clubs in area")
                ], className = "three columns"
            ),
            html.Div(style={"backgroundColor": "#5D6D7E", "textAlign": "center"}, children=
                [
                    html.Div(id = "no_restaurant_area"),
                    html.P("# of restaurants in area")
                ], className = "three columns"
            ),
                
        ], style=kpi_style, className = "row"
    ),
    # Create map scatter and datatable next to it
    html.Div(children=
             [
                 html.Div([
                     dcc.Graph(id="bar_scatter")
                 ], style=kpi_style, className="six columns"
                 ),
                 html.Div([
                     dcc.Graph(id="license_bar")
                 ], style=kpi_style, className="six columns"
                 )
             ], style=kpi_style, className="row"
    )
], className = "row"
)


# # Model page

# In[13]:


from sklearn.preprocessing import OneHotEncoder
type_ohe = OneHotEncoder()
suburb_ohe = OneHotEncoder()

# One hot encode Type
df_type = pd.DataFrame(type_ohe.fit_transform(df_merge['Type'].values.reshape(-1,1)).toarray())
df_type.columns = type_ohe.get_feature_names(['Type'])
df_type = df_type.rename({"Type_u": "Type_Unit", "Type_h": "Type_House", "Type_t": "Type_Townhouse"}, axis=1)

# One hot encode geom_suburb
df_geomsuburb = pd.DataFrame(suburb_ohe.fit_transform(df_merge['geom_suburb'].values.reshape(-1,1)).toarray())
df_geomsuburb.columns = suburb_ohe.get_feature_names(['Suburb'])


# In[14]:


y = df_merge.Price.values
X = pd.concat([df_merge[['Rooms', 'Bathroom', "Car", "Properties per Bar", "Distance", "suburb area", "Landsize"]], 
               df_type, df_geomsuburb], axis=1)


# In[15]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_fit = scaler.fit_transform(X)


# In[16]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, max_depth=17)
reg = rf.fit(X_fit, y)


# In[17]:


# Create input
X_filter = pd.concat([df_merge[['Bedroom', 'Bathroom', "Car", "Properties per Bar", "Distance", "suburb area", "Landsize"]], 
       df_type, df_geomsuburb], axis=1)
y_pred = rf.predict(X_filter)
df_merge['Suggested Price'] = y_pred.round(2)


# In[18]:


sorted_idx = reg.feature_importances_.argsort()[-10:]
sorted_features, sorted_imp = [x.title() for x in X.columns[sorted_idx]], reg.feature_importances_[sorted_idx]


# In[19]:


rank_table_columns = ["Importance (%)", "Feature"]
rank_table = pd.DataFrame([sorted_imp, sorted_features]).transpose()
rank_table.columns = rank_table_columns
rank_table['Importance (%)'] = rank_table['Importance (%)']*100
rank_table = rank_table.sort_values(by="Importance (%)", ascending=False).reset_index(drop=True)
rank_table['Importance (%)'] = round(rank_table['Importance (%)'].astype(float),1)


# In[23]:


model_layout = html.Div(style=app_style,children=[
    # Title - Row
    html.Div(
        [
            html.H1(
                'Price prediction',
                className='five columns',
            ),
            html.Div([ 
                dcc.Link('Home', href='/', style = link_style)
                    ], className = 'one columns', style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Div([
                dcc.Link('General Statistics', href='/general_statistics', style=link_style) 
            ], className = "one columns", style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Div([
                dcc.Link('Housing overview', href='/housing_overview', style = link_style) 
            ], className = "one columns", style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Div([
                dcc.Link('Area niceness', href='/area_niceness', style = link_style) 
            ], className = "one columns", style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Div([
                dcc.Link('Price prediction', href='/model_page', style = link_style) 
            ], className = "one columns", style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Img(
                src="http://logok.org/wp-content/uploads/2014/06/City-of-Melbourne-logo-M-880x660.png",
                className='two columns',
                style={
                    'height': '12%',
                    'width': '12%',
                    'float': 'right',
                    'position': 'top',
                    'padding-top': 0,
                    'padding-right': 0
                },
            ),
            html.P('Set the filters to match your personal preferences, the model below will then show the houses for sale that matches your choices. In the table on the right, you can see the actual selling price compared to the predicted price to evaluate the potential sale. At the bottom of the page, you can find how much the different features influence the price prediction',
                className='eight columns',
            ),
        ],
        className='row'
    ),
    # Selectors
    html.Div(
        [
            html.Div(
                [                    
                    html.P('Suburb:'),
                    dcc.Dropdown(
                        id='suburb_drop_model',
                        options= suburb_class,
                        multi=False,
                        style={"color": "#616A6B"}
                    )
                ],
                className='two columns',
                style={'margin-top': '10'}
            ),
            html.Div(
                [                    
                    html.P('Housing Type:'),
                    dcc.Dropdown(
                        id='type_drop_model',
                        options= housing_class,
                        multi=False,
                        style={"color": "#616A6B"}
                    )
                ],
                className='two columns',
                style={'margin-top': '10'}
            ),
            html.Div(
                [                    
                    html.P('Distance to CBD:'),
                    dcc.RangeSlider(
                        id='distance_slider_model',
                        min = 0,
                        max = int(math.ceil(housing.Distance.max() / 10.0))*10,
                        marks = distance_class,
                        value=[0, int(math.ceil(housing.Distance.max() / 10.0))*10],
                    )
                ],
                className='two columns', style={'margin-top': '10'}
            ),
            html.Div(
                [                    
                    html.P('Car:'),
                    dcc.RangeSlider(
                        id='car_slider_model',
                        min = int(df_merge.Car.min()),
                        max = int(df_merge.Car.max()),
                        marks = car_class,
                        value=[int(df_merge.Car.min()), int(df_merge.Car.max())],
                    )
                ],
                className='two columns',
                style={'margin-top': '10'}
            ),
            html.Div(
                [                    
                    html.P('Rooms:'),
                    dcc.RangeSlider(
                        id='rooms_slider_model',
                        min = housing.Rooms.min(),
                        max = housing.Rooms.max(),
                        marks = rooms_class,
                        value=[housing.Rooms.min(), housing.Rooms.max()],
                    )
                ],
                className='two columns',
                style={'margin-top': '10'}
            ),
            html.Div(
                [                    
                    html.P('Bathrooms:'),
                    dcc.RangeSlider(
                        id='bathrooms_slider_model',
                        min = int(df_merge.Bathroom.min()),
                        max = int(df_merge.Bathroom.max()),
                        marks = bathroom_class,
                        value=[int(df_merge.Bathroom.min()), int(df_merge.Bathroom.max())],
                    )
                ],
                className='two columns',
                style={'margin-top': '10'}
            )
        ], style=subdiv_style, className='row'
    ),
    
    # Create map scatter
    html.Div(children=
             [
                 html.Div([
                     dcc.Graph(id="map_scatter")
                 ], className="eight columns"
                 ),
                 html.Div([
                     html.P("Below you can see which features have the biggest influence on the price.  These features can be helpful in case the suggested houses does not match your current budget. As you can see what features will give the biggest payoff if you change e.g. from 5 to 3 rooms. Play around with the filters to find your perfect match."),
                     dash_table.DataTable(columns = [{"name": i, "id": i} for i in rank_table.columns],
                     data = rank_table.to_dict("records"),
                     style_cell = {"textAlign" : "left", "maxWidth": "100px", "minWidth": "50px"},
                     style_header={"backgroundColor":"#5D6D7E"},
                     style_data = {"backgroundColor": "rgba(41,47,53,1)"},
                     style_table = {"overflowY":"auto", "height": 500},
                     fixed_rows = {"headers": True})
                 ], className = "four columns"
                 )
             ], style={"padding": 0, "margin": 0}, className="row"
    ),
    # Add datatable
    html.Div(children=
             [
                 html.P("The properties in the below table match the selected criteria. On the right side of the table you can see the actual price and the predicted price of the property. This gives an indication of whether the property is below or above the marketprice."),
                 html.Div(id = "datatable")
             ], style = {"padding": 0, "margin": 0}, className="twelve columns"
            )
], className = "row"
)


# # Total app

# In[24]:


import dash
import dash_core_components as dcc
import dash_html_components as html

# Since we're adding callbacks to elements that don't exist in the app.layout,
# Dash will raise an exception to warn us that we might be
# doing something wrong.
# In this case, we're adding the elements through a callback, so we can ignore
# the exception.
ext_style = ['https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css']
app = dash.Dash(__name__, external_stylesheets = ext_style, suppress_callback_exceptions = True)
server = app.server
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


index_page = html.Div([
    html.Div(
        [
            html.H1(
                'Melbourne Housing Platform',
                className='five columns',
            ),
            html.Div([ 
                dcc.Link('Home', href='/', style = link_style)
                    ], className = 'one columns', style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Div([
                dcc.Link('General Statistics', href='/general_statistics', style=link_style) 
            ], className = "one columns", style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Div([
                dcc.Link('Housing overview', href='/housing_overview', style = link_style) 
            ], className = "one columns", style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Div([
                dcc.Link('Area niceness', href='/area_niceness', style = link_style) 
            ], className = "one columns", style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Div([
                dcc.Link('Price prediction', href='/model_page', style = link_style) 
            ], className = "one columns", style = {"backgroundColor": "#016712", "textAlign": "center",
                                                          "padding-bottom": 10, "padding-top": 10,
                                                          "padding-left": 5, "padding-right": 5}
            ),
            html.Img(
                src="http://logok.org/wp-content/uploads/2014/06/City-of-Melbourne-logo-M-880x660.png",
                className='two columns',
                style={
                    'height': '12%',
                    'width': '12%',
                    'float': 'right',
                    'position': 'top',
                    'padding-top': 0,
                    'padding-right': 0
                },
            ),
        ],
        className='row'
    ),
    html.Div([
                html.P("The explainer notebook can be downloaded by following the link below"),
                html.A("Explainer notebook", href="https://www.dropbox.com/s/h4gbt5rx2dmymws/Explainer%20notebook.ipynb?dl=0")
            ]),
            html.Img(
            src = app.get_asset_url("vision.png"),
            className = "eight columns"
    ),
    html.Div([
        html.Img(
            src = app.get_asset_url("Frontpage.png"),
            className = "twelve columns",
            style = {
                "height": "80%",
                "width": "80%",
                "padding-left": 25
            }
        )
    ], className = "row"
    )
], style=app_style)

######################## General statistics callbacks #############################
@app.callback(
    [
        Output('avg_price_stats', 'children'),
        Output('no_listings_stats', 'children'),
        Output("no_bars_stats", "children"),
        Output("no_properties_stats", "children")
    ],
    Input('suburb_drop_stats', 'value')
)
def update_kpis(suburb_select):
    if suburb_select in df_merge.geom_suburb.unique().tolist():
        temp_df = df_merge[df_merge.geom_suburb == suburb_select].copy()
    else:
        temp_df = df_merge.copy()

    return f"{round(temp_df.Price.mean()/10**6, 2)} M", len(temp_df), sum(temp_df.Bars.unique().tolist()), round(sum(temp_df.Propertycount.unique().tolist()))

@app.callback([
    Output("grid1_1", "figure"), 
    Output("grid1_2", "figure"), 
    Output("grid1_3", "figure"), 
    Output("grid1_4", "figure"),
    Output("grid1_5", "figure"),
    Output("grid1_6", "figure"),
    Output("grid2_1", "figure"), 
    Output("grid2_2", "figure"), 
    Output("grid2_3", "figure"), 
    Output("grid2_4", "figure"),
    Output("grid2_5", "figure"),
    Output("grid2_6", "figure"),
    Output("grid3_1", "figure"), 
    Output("grid3_2", "figure"), 
    Output("grid3_3", "figure"), 
    Output("grid3_4", "figure"),
    Output("grid3_5", "figure"),
    Output("grid3_6", "figure")],
    Input("suburb_drop_stats", "value")
)
def create_housing_row(suburb_select):
    if suburb_select in df_merge.geom_suburb.unique().tolist():
        temp_df = df_merge[df_merge.geom_suburb == suburb_select].copy()
        df_house = temp_df[temp_df.Type == "h"].copy()
        df_unit = temp_df[temp_df.Type == "u"].copy()
        df_town = temp_df[temp_df.Type == "t"].copy()
    else:
        df_house = df_merge[df_merge.Type == "h"].copy()
        df_unit = df_merge[df_merge.Type == "u"].copy()
        df_town = df_merge[df_merge.Type == "t"].copy()
        
    # House row   
    fig1_1 = px.scatter(df_house, x = "Distance", y = "Price", color_discrete_sequence=["cyan"], render_mode = "svg")
    fig1_2 = px.scatter(df_house, x = "Rooms", y = "Price", color_discrete_sequence=["cyan"], render_mode = "svg")
    fig1_3 = px.scatter(df_house, x = "Bathroom", y = "Price", color_discrete_sequence=["cyan"], render_mode = "svg")
    fig1_4 = px.scatter(df_house, x = "Car", y = "Price", color_discrete_sequence=["cyan"], render_mode = "svg")
    fig1_5 = px.scatter(df_house, x = "Properties per Bar", y = "Price", color_discrete_sequence=["cyan"], render_mode = "svg")
    fig1_6 = px.scatter(df_house, x = "suburb area", y = "Price", color_discrete_sequence=["cyan"], render_mode = "svg")    
    fig1_1['layout'].update(margin=dict(l=0,r=5,b=0,t=5), plot_bgcolor= 'rgba(93,109,126,1)', paper_bgcolor = 'rgba(41,47,53,1)', xaxis={"visible": True, 'title': None, "color": "white"}, yaxis={"visible": True, 'title': "House Price", "color": "white", "titlefont": {"size": 11}}, height=200)
    fig1_2['layout'].update(margin=dict(l=0,r=5,b=0,t=5), plot_bgcolor= 'rgba(93,109,126,1)', paper_bgcolor = 'rgba(41,47,53,1)', xaxis={"visible": True, 'title': None, "color": "white"}, yaxis={"visible": True, 'title': None, "color": "white"}, height=200)
    fig1_3['layout'].update(margin=dict(l=0,r=5,b=0,t=5), plot_bgcolor= 'rgba(93,109,126,1)', paper_bgcolor = 'rgba(41,47,53,1)', xaxis={"visible": True, 'title': None, "color": "white"}, yaxis={"visible": True, 'title': None, "color": "white"}, height=200)
    fig1_4['layout'].update(margin=dict(l=0,r=5,b=0,t=5), plot_bgcolor= 'rgba(93,109,126,1)', paper_bgcolor = 'rgba(41,47,53,1)', xaxis={"visible": True, 'title': None, "color": "white"}, yaxis={"visible": True, 'title': None, "color": "white"}, height=200)
    fig1_5['layout'].update(margin=dict(l=0,r=5,b=0,t=5), plot_bgcolor= 'rgba(93,109,126,1)', paper_bgcolor = 'rgba(41,47,53,1)', xaxis={"visible": True, 'title': None, "color": "white"}, yaxis={"visible": True, 'title': None, "color": "white"}, height=200)
    fig1_6['layout'].update(margin=dict(l=0,r=5,b=0,t=5), plot_bgcolor= 'rgba(93,109,126,1)', paper_bgcolor = 'rgba(41,47,53,1)', xaxis={"visible": True, 'title': None, "color": "white"}, yaxis={"visible": True, 'title': None, "color": "white"}, height=200)
    
    # Unit row
    fig2_1 = px.scatter(df_unit, x = "Distance", y = "Price", render_mode = "svg")
    fig2_2 = px.scatter(df_unit, x = "Rooms", y = "Price", render_mode = "svg")
    fig2_3 = px.scatter(df_unit, x = "Bathroom", y = "Price", render_mode = "svg")
    fig2_4 = px.scatter(df_unit, x = "Car", y = "Price", render_mode = "svg")
    fig2_5 = px.scatter(df_unit, x = "Properties per Bar", y = "Price", render_mode = "svg")
    fig2_6 = px.scatter(df_unit, x = "suburb area", y = "Price", render_mode = "svg")    
    fig2_1['layout'].update(margin=dict(l=0,r=5,b=0,t=5), plot_bgcolor= 'rgba(93,109,126,1)', paper_bgcolor = 'rgba(41,47,53,1)', xaxis={"visible": True, 'title': None, "color": "white"}, yaxis={"visible": True, 'title': "Unit Price", "color": "white", "titlefont": {"size": 11}}, height=200)
    fig2_2['layout'].update(margin=dict(l=0,r=5,b=0,t=5), plot_bgcolor= 'rgba(93,109,126,1)', paper_bgcolor = 'rgba(41,47,53,1)', xaxis={"visible": True, 'title': None, "color": "white"}, yaxis={"visible": True, 'title': None, "color": "white"}, height=200)
    fig2_3['layout'].update(margin=dict(l=0,r=5,b=0,t=5), plot_bgcolor= 'rgba(93,109,126,1)', paper_bgcolor = 'rgba(41,47,53,1)', xaxis={"visible": True, 'title': None, "color": "white"}, yaxis={"visible": True, 'title': None, "color": "white"}, height=200)
    fig2_4['layout'].update(margin=dict(l=0,r=5,b=0,t=5), plot_bgcolor= 'rgba(93,109,126,1)', paper_bgcolor = 'rgba(41,47,53,1)', xaxis={"visible": True, 'title': None, "color": "white"}, yaxis={"visible": True, 'title': None, "color": "white"}, height=200)
    fig2_5['layout'].update(margin=dict(l=0,r=5,b=0,t=5), plot_bgcolor= 'rgba(93,109,126,1)', paper_bgcolor = 'rgba(41,47,53,1)', xaxis={"visible": True, 'title': None, "color": "white"}, yaxis={"visible": True, 'title': None, "color": "white"}, height=200)
    fig2_6['layout'].update(margin=dict(l=0,r=5,b=0,t=5), plot_bgcolor= 'rgba(93,109,126,1)', paper_bgcolor = 'rgba(41,47,53,1)', xaxis={"visible": True, 'title': None, "color": "white"}, yaxis={"visible": True, 'title': None, "color": "white"}, height=200)
    
    
    # Townhouse row
    fig3_1 = px.scatter(df_town, x = "Distance", y = "Price", color_discrete_sequence=["pink"], render_mode = "svg")
    fig3_2 = px.scatter(df_town, x = "Rooms", y = "Price", color_discrete_sequence=["pink"], render_mode = "svg")
    fig3_3 = px.scatter(df_town, x = "Bathroom", y = "Price", color_discrete_sequence=["pink"], render_mode = "svg")
    fig3_4 = px.scatter(df_town, x = "Car", y = "Price", color_discrete_sequence=["pink"], render_mode = "svg")
    fig3_5 = px.scatter(df_town, x = "Properties per Bar", y = "Price", color_discrete_sequence=["pink"], render_mode = "svg")
    fig3_6 = px.scatter(df_town, x = "suburb area", y = "Price", color_discrete_sequence=["pink"], render_mode = "svg")    
    fig3_1['layout'].update(margin=dict(l=0,r=5,b=0,t=5), plot_bgcolor= 'rgba(93,109,126,1)', paper_bgcolor = 'rgba(41,47,53,1)', xaxis={"visible": True, 'title': None, "color": "white"}, yaxis={"visible": True, 'title': "Townhouse price", "color": "white", "titlefont": {"size": 11}}, height=200)
    fig3_2['layout'].update(margin=dict(l=0,r=5,b=0,t=5), plot_bgcolor= 'rgba(93,109,126,1)', paper_bgcolor = 'rgba(41,47,53,1)', xaxis={"visible": True, 'title': None, "color": "white"}, yaxis={"visible": True, 'title': None, "color": "white"}, height=200)
    fig3_3['layout'].update(margin=dict(l=0,r=5,b=0,t=5), plot_bgcolor= 'rgba(93,109,126,1)', paper_bgcolor = 'rgba(41,47,53,1)', xaxis={"visible": True, 'title': None, "color": "white"}, yaxis={"visible": True, 'title': None, "color": "white"}, height=200)
    fig3_4['layout'].update(margin=dict(l=0,r=5,b=0,t=5), plot_bgcolor= 'rgba(93,109,126,1)', paper_bgcolor = 'rgba(41,47,53,1)', xaxis={"visible": True, 'title': None, "color": "white"}, yaxis={"visible": True, 'title': None, "color": "white"}, height=200)
    fig3_5['layout'].update(margin=dict(l=0,r=5,b=0,t=5), plot_bgcolor= 'rgba(93,109,126,1)', paper_bgcolor = 'rgba(41,47,53,1)', xaxis={"visible": True, 'title': None, "color": "white"}, yaxis={"visible": True, 'title': None, "color": "white"}, height=200)
    fig3_6['layout'].update(margin=dict(l=0,r=5,b=0,t=5), plot_bgcolor= 'rgba(93,109,126,1)', paper_bgcolor = 'rgba(41,47,53,1)', xaxis={"visible": True, 'title': None, "color": "white"}, yaxis={"visible": True, 'title': None, "color": "white"}, height=200)
    
    return fig1_1, fig1_2, fig1_3, fig1_4, fig1_5, fig1_6, fig2_1, fig2_2, fig2_3, fig2_4, fig2_5, fig2_6, fig3_1, fig3_2, fig3_3, fig3_4, fig3_5, fig3_6


######################## Housing callbacks #############################
@app.callback(
    [
    Output('avg_price_housing', 'children'),
    Output('no_listings_housing', 'children'),
    Output('no_bars_housing', 'children'),
    Output('no_distance_housing', 'children')],
    [
    Input('suburb_drop_housing', 'value'),
    Input('type_drop_housing', 'value'),
    Input('rooms_slider_housing', 'value'),
    Input('distance_slider_housing', 'value')]
)
def update_kpi_labels(suburb_select, type_select, rooms_select, distance_select):
    if suburb_select in df_merge.geom_suburb.unique().tolist():
        temp_df = df_merge[df_merge.geom_suburb == suburb_select]
    else:
        temp_df = df_merge.copy()
    
    if type_select in df_merge.Type.unique().tolist():
        temp_df = temp_df[temp_df.Type == type_select]
    else:
        temp_df = temp_df.copy()
        
    temp_df = temp_df[(temp_df.Rooms >= rooms_select[0]) & (temp_df.Rooms <= rooms_select[1])].copy()
    temp_df = temp_df[(temp_df.Distance >= distance_select[0]) & (temp_df.Distance <= distance_select[1])].copy()
    
    avg_price = "N/A" if math.isnan(temp_df.Price.mean()/10**6) else round(temp_df.Price.mean()/10**6,2)
    listings = "N/A" if math.isnan(len(temp_df)) else len(temp_df)
    bars = "N/A" if math.isnan(temp_df.Bars.mean()) else round(temp_df.Bars.mean())
    distance = "N/A" if math.isnan(temp_df.Distance.mean()) else round(temp_df.Distance.mean())
    return f"{avg_price} M", listings, bars, f"{distance} km"

@app.callback(
    Output('choropleth_map_housing', 'figure'),
    [
    Input('suburb_drop_housing', 'value'),
    Input('type_drop_housing', 'value'),
    Input('rooms_slider_housing', 'value'),
    Input('distance_slider_housing', 'value')]
)
def update_choropleth(suburb_select, type_select, rooms_select, distance_select):
    if suburb_select in df_merge.geom_suburb.unique().tolist():
        temp_df = df_merge[df_merge.geom_suburb == suburb_select]
    else:
        temp_df = df_merge.copy()
    
    if type_select in df_merge.Type.unique().tolist():
        temp_df = temp_df[temp_df.Type == type_select]
    else:
        temp_df = temp_df.copy()
        
    temp_df = temp_df[(temp_df.Rooms >= rooms_select[0]) & (temp_df.Rooms <= rooms_select[1])].copy()
    temp_df = temp_df[(temp_df.Distance >= distance_select[0]) & (temp_df.Distance <= distance_select[1])].copy()
        
    choropleth_df = pd.merge(left=shapes[['NAME_2', "geometry"]], 
                             right=temp_df.groupby(['Regionname', 'geom_suburb', 'suburb area']).agg(["mean", "count"])['Price'].reset_index(), 
                             how='right', left_on = "NAME_2", right_on = "geom_suburb")
    choropleth_df = choropleth_df.drop('NAME_2', axis=1)
    choropleth_df['mean'] = round(choropleth_df['mean'] / (1*10**6),1)
    choropleth_df['suburb area'] = round(choropleth_df['suburb area'], 4)
    choropleth_df = choropleth_df.rename({"mean": "Price", "count": "Listings"}, axis=1)

    merged_json = json.loads(choropleth_df.to_json())
    
    fig = px.choropleth_mapbox(choropleth_df,
                           geojson = merged_json,featureidkey = "properties.geom_suburb",
                           locations="geom_suburb",
                           color='Price',
                           color_continuous_scale="Viridis_r",
                           range_color = (0, 2),
                           hover_name = "geom_suburb",
                           hover_data = ["Price", "geom_suburb", "suburb area", "Listings", "Regionname"],
                           labels= {
                               "Price": "Price (mAUD)",
                               "geom_suburb": "Suburb",
                               "suburb area": "Suburb Area",
                               "Listings": "Housing in Area",
                               "Regionname": "Region"
                           },
                           mapbox_style = "carto-darkmatter",
                           zoom=8,
                           center={"lat": -37.8, "lon": 144.95},
                          )

    # Define layout specificities
    fig.update_layout(
        margin={'r':0,'t':0,'l':0,'b':0},
        coloraxis_colorbar={
            'title':'Price (mAUD)'       
        }, 
        paper_bgcolor = 'rgba(41,47,53,1)',
        font_color = "white",
        coloraxis_colorbar_x = -0.15,
        title = {"text": "Melbourne suburbs colored according to avg. housing price", "x": 0.5, "y": 0.99, 
                 "font":{"size": 14}}
    
    )
    
    return fig

@app.callback(
    Output('distance_hbar', 'figure'),
    [
    Input('type_drop_housing', 'value'),
    Input('rooms_slider_housing', 'value'),
    Input('distance_slider_housing', 'value')]
)
def update_avg_dist(type_select, rooms_select, distance_select):
    if type_select in df_merge.Type.unique().tolist():
        temp_df = df_merge[df_merge.Type == type_select]
    else:
        temp_df = df_merge.copy()
        
    temp_df = temp_df[(temp_df.Rooms >= rooms_select[0]) & (temp_df.Rooms <= rooms_select[1])].copy()
    temp_df = temp_df[(temp_df.Distance >= distance_select[0]) & (temp_df.Distance <= distance_select[1])].copy()
    
    temp_df = temp_df.groupby(["geom_suburb"]).mean()[['Distance','Price']].sort_values(by="Distance", ascending=True).reset_index()
    temp_df['Price'] = round(temp_df['Price'] / (1*10**6),1)
    fig = px.bar(temp_df, x="geom_suburb", y="Distance", range_color = (0,2), 
                 color="Price", color_continuous_scale = "Viridis_r", hover_data = ["Price"],
            labels={
                     "geom_suburb": "Suburb"
                 })

    fig['layout'].update(margin=dict(l=0,r=5,b=0,t=5), plot_bgcolor= 'rgba(93,109,126,1)', 
                         paper_bgcolor = 'rgba(41,47,53,1)', 
                 yaxis={"visible": True, 'title': "Avg. Distance to CBD (km)", "color": "white"}, 
                 xaxis={"visible": True, 'title': None, "color": "white", "tickfont": {"size": 10}})
    fig.update(layout_coloraxis_showscale=False)
    fig.update_yaxes(categoryorder = "total descending")
    return fig

@app.callback(
    Output('housing_price_bar', 'figure'),
    [
    Input('suburb_drop_housing', 'value'),
    Input('type_drop_housing', 'value'),
    Input('rooms_slider_housing', 'value'),
    Input('distance_slider_housing', 'value')]
)
def update_housing_price(suburb_select, type_select, rooms_select, distance_select):
    if suburb_select in df_merge.geom_suburb.unique().tolist():
        temp_df = df_merge[df_merge.geom_suburb == suburb_select]
    else:
        temp_df = df_merge.copy()
    
    if type_select in df_merge.Type.unique().tolist():
        temp_df = temp_df[temp_df.Type == type_select]
    else:
        temp_df = temp_df.copy()
        
    temp_df = temp_df.groupby(["geom_suburb", "Type"]).mean()['Price'].reset_index()
    temp_df = temp_df.replace("h", "Home")
    temp_df = temp_df.replace("u", "Unit")
    temp_df = temp_df.replace("t", "Townhouse")

    fig = px.bar(temp_df, y="Price", x="geom_suburb", color="Type", barmode="group", 
             color_discrete_sequence = ['#40E0D0', "#6495ED", "#CCCCFF"]*df_merge.geom_suburb.nunique(),
            labels={
                     "Type": ""
                 })
    fig['layout'].update(margin=dict(l=0,r=5,b=0,t=5), plot_bgcolor= 'rgba(93,109,126,1)', paper_bgcolor = 'rgba(41,47,53,1)', 
                     xaxis={"visible": True, 'title': None, "color": "white"}, 
                     yaxis={"visible": True, 'title': None, "color": "white"},
                    legend={"orientation": "h", "yanchor": "top", "xanchor":"right", "y": 1.05, "x":1,
                            "font": {"color": "white"}},
                    title = {"text": "Housing price per type and suburb", "x": 0.5, "y": 0.95, 
                 "font":{"size": 14, "color": "white"}})
    return fig

################### Area niceness ###############################3
@app.callback(
    [
    Output('no_bars_area', 'children'),
    Output('no_propbar_area', 'children'),
    Output('no_nightclub_area', 'children'),
    Output('no_restaurant_area', 'children')],
    [
    Input('suburb_drop_area', 'value'),
    Input('category_drop_area', 'value'),
    Input('trading_drop_area', 'value')
    ]
)
def update_kpi_labels(suburb_select, category_select, trading_select):
    if suburb_select in licenses.geom_suburb.unique().tolist():
        temp_df = licenses[licenses.geom_suburb == suburb_select]
        temp_merge = df_merge[df_merge.geom_suburb == suburb_select]
    else:
        temp_df = licenses.copy()
        temp_merge = df_merge.copy()
        
    if category_select in licenses.Category.unique().tolist():
        temp_df = temp_df[temp_df.Category == category_select]
    else:
        temp_df = temp_df.copy()
    
    if trading_select:
        temp_df = temp_df[temp_df['Trading mapping'] >= int(trading_select)].reset_index(drop=True)
    else:
        temp_df = temp_df.copy()
    
    tot_bars = "N/A" if len(temp_df) < 1 else len(temp_df)
    propbars = "N/A" if math.isnan(temp_merge['Properties per Bar'].mean()) else round(temp_merge['Properties per Bar'].mean(), 2)
    nightclubs = "N/A" if len(temp_df[temp_df['Trading mapping'] >= 4]) < 1 else len(temp_df[temp_df['Trading mapping'] >= 4])
    restaurants = "N/A" if len(temp_df[(temp_df['Trading mapping'] < 4) & (temp_df['Trading mapping'] > 0)]) < 1 else len(temp_df[(temp_df['Trading mapping'] <= 4) & (temp_df['Trading mapping'] > 0)])
    return tot_bars, propbars, nightclubs, restaurants


@app.callback(
    Output('bar_scatter', 'figure'),
    [
    Input('suburb_drop_area', 'value'),
    Input('category_drop_area', 'value'),
    Input('trading_drop_area', 'value')
    ]
)
def update_scatter(suburb_select, category_select, trading_select):
    zoom = 8
    if suburb_select in licenses.geom_suburb.unique().tolist():
        temp_df = licenses[licenses.geom_suburb == suburb_select]
        zoom = 11
    else:
        temp_df = licenses.copy()
        
    if category_select in licenses.Category.unique().tolist():
        temp_df = temp_df[temp_df.Category == category_select]
    else:
        temp_df = temp_df.copy()
    
    if trading_select:
        temp_df = temp_df[temp_df['Trading mapping'] >= int(trading_select)].reset_index(drop=True)
    else:
        temp_df = temp_df.copy()
    
    temp_df['Bar_type'] = ["Retailer" if i in ['Bottle shop', 'Wholesaler'] 
                           else "Bars & Restaurants" for i in temp_df['After 11 pm']]
    colors_map = {"Retailer": "#FDB23A", "Bars & Restaurants": "#FA83F0"}
    
    px.set_mapbox_access_token('pk.eyJ1IjoiamVzcGVyaGF1Y2giLCJhIjoiY2tvYWR2MXUxMDVqZjJ3b2R6a2M2eDJ5ZSJ9.I1Ld1d4B1756mz4bhSWTqw')
    fig = px.scatter_mapbox(temp_df, lat = "Latitude", 
                            lon = "Longitude",
                            color = "Bar_type",
                            color_discrete_map = colors_map,
                            opacity = 0.8,
                            hover_name = 'Trading As',
                            hover_data = {"Latitude": False, "Longitude": False, 
                                          "Category": True, "geom_suburb" : True, 
                                          "After 11 pm": True},
                            mapbox_style = "carto-darkmatter",
                            labels = {
                                "geom_suburb": "Suburb",
                                "Bar_type": "Type",
                                "After 11 pm": "Opening Hours"
                            },
                            zoom = zoom)
                            

    # Define layout specificities
    fig.update_layout(
        margin={'r':0,'t':0,'l':0,'b':0}, 
        paper_bgcolor = 'rgba(41,47,53,1)',
        font_color = "white",
        showlegend=False,
        title = {"text": "Liquor license holders colored according to license type", "x": 0.5, "y": 0.99, 
                 "font":{"size": 14}}
    )
    
    return fig

@app.callback(
    Output('license_bar', 'figure'),
    [
    Input('suburb_drop_area', 'value')]
)
def update_bar(suburb_select):
    if suburb_select in licenses.geom_suburb.unique().tolist():
        temp_df = licenses[licenses.geom_suburb == suburb_select]
    else:
        temp_df = licenses.copy()
        
    temp_df = temp_df.groupby("After 11 pm").count()['Trading Hours'].reset_index().sort_values(by="After 11 pm")
    colors = ["#FDB23A" if i in ['Bottle shop', 'Wholesaler'] else "#FA83F0" for i in temp_df['After 11 pm']]
    temp_df['Bar_type'] = ["Retailer" if i == "#FDB23A" else "Bars & Restaurants" for i in colors]
    
    fig = px.bar(temp_df, x="After 11 pm", y="Trading Hours", color="Bar_type", color_discrete_sequence = colors,
                labels = {
                    "Bar_type": "Type",
                    "After 11 pm": "Trading hours",
                    "Trading Hours": "Value"
                            })

    fig['layout'].update(margin=dict(l=0,r=5,b=0,t=5), plot_bgcolor= 'rgba(93,109,126,1)', 
                         paper_bgcolor = 'rgba(41,47,53,1)', 
                 yaxis={"visible": True, 'title': "Number of establishments", "color": "white"}, 
                 xaxis={"visible": True, 'title': None, "color": "white"},
                 legend={"font": {"color": "white"}, "title": "License holder type", 
                         "yanchor": "top", "xanchor":"right", "x":1, "title_font_color": "white"})
    fig.update_yaxes(categoryorder = "total descending")
    return fig

########################### Model callbacks ###########################
@app.callback(
    Output('map_scatter', 'figure'),
    [
    Input('suburb_drop_model', 'value'),
    Input('distance_slider_model', 'value'),
    Input('type_drop_model', 'value'),
    Input('car_slider_model', 'value'),
    Input('rooms_slider_model', 'value'),
    Input('bathrooms_slider_model', 'value')]
)
def update_scatter(suburb_select, distance_interval, type_select, car_interval, rooms_interval, bathrooms_interval):
    zoom = 8
    if suburb_select in df_merge.geom_suburb.unique().tolist():
        temp_df = df_merge[df_merge.geom_suburb == suburb_select]
        zoom=11
    else:
        temp_df = df_merge.copy()
    
    temp_df = temp_df[(temp_df.Distance >= distance_interval[0]) & (temp_df.Distance <= distance_interval[1])].copy()
    
    if type_select in df_merge.Type.unique().tolist():
        temp_df = temp_df[temp_df.Type == type_select]
    else:
        temp_df = temp_df.copy()
    
    temp_df = temp_df[(temp_df.Car >= car_interval[0]) & (temp_df.Car <= car_interval[1])].copy()
    temp_df = temp_df[(temp_df.Rooms >= rooms_interval[0]) & (temp_df.Rooms <= rooms_interval[1])].copy()
    temp_df = temp_df[(temp_df.Bathroom >= bathrooms_interval[0]) & (temp_df.Bathroom <= bathrooms_interval[1])].copy()
    
    temp_df = temp_df.replace({"Type": {"h": "Home", "u": "Unit", "t": "Townhouse"}})  
    px.set_mapbox_access_token('pk.eyJ1IjoiamVzcGVyaGF1Y2giLCJhIjoiY2tvYWR2MXUxMDVqZjJ3b2R6a2M2eDJ5ZSJ9.I1Ld1d4B1756mz4bhSWTqw')
    fig = px.scatter_mapbox(temp_df, lat = temp_df.Latitude, 
                            lon = temp_df.Longitude, 
                            hover_name = "Address", 
                            hover_data = {"Latitude": False, "Longitude": False,
                                          "Price": True, "geom_suburb": True,
                                          "Type": True, "Rooms": True, 
                                          "Bathroom": True, "Landsize": True,
                                          "Car": True, "Distance": True},
                            labels = {
                                "Distance": "Distance to CBD (km)",
                                "Type": "Housing Type",
                                "geom_suburb": "Suburb"},
                            mapbox_style = "carto-darkmatter",
                            height=500,
                            zoom = zoom)
                            

    # Define layout specificities
    fig.update_layout(
        margin={'r':0,'t':0,'l':0,'b':0}, 
        paper_bgcolor = 'rgba(41,47,53,1)',
        font_color = "white",
        title = {"text": "Houses matching selected criteria", "x": 0.5, "y": 0.99, 
                 "font":{"size": 14, "color": "white"}}
    )
    
    return fig

@app.callback(
    Output('datatable', 'children'),
    [
    Input('suburb_drop_model', 'value'),
    Input('distance_slider_model', 'value'),
    Input('type_drop_model', 'value'),
    Input('car_slider_model', 'value'),
    Input('rooms_slider_model', 'value'),
    Input('bathrooms_slider_model', 'value')
    ]
)
def update_table(suburb_select, distance_interval, type_select, car_interval, rooms_interval, bathrooms_interval):
    if suburb_select in df_merge.geom_suburb.unique().tolist():
        temp_df = df_merge[df_merge.geom_suburb == suburb_select]
    else:
        temp_df = df_merge.copy()
    
    temp_df = temp_df[(temp_df.Distance >= distance_interval[0]) & (temp_df.Distance <= distance_interval[1])].copy()
    
    if type_select in df_merge.Type.unique().tolist():
        temp_df = temp_df[temp_df.Type == type_select]
    else:
        temp_df = temp_df.copy()
    
    temp_df = temp_df[(temp_df.Car >= car_interval[0]) & (temp_df.Car <= car_interval[1])].copy()
    temp_df = temp_df[(temp_df.Rooms >= rooms_interval[0]) & (temp_df.Rooms <= rooms_interval[1])].copy()
    temp_df = temp_df[(temp_df.Bathroom >= bathrooms_interval[0]) & (temp_df.Bathroom <= bathrooms_interval[1])].copy()
    
    temp_df = temp_df.replace({"Type": {"h": "House", "u": "Unit", "t": "Townhouse"}})
    temp_df = temp_df[["Address", "geom_suburb", "Distance", "Type", 
                       "Landsize", "Car", "Rooms", "Bathroom", "Price", "Suggested Price"]]
    temp_df = temp_df.rename({"Type": "Housing Type", 
                              "geom_suburb": "Suburb",
                              "Distance": "Distance to CBD"}, axis=1)
    temp_df = temp_df.sort_values(by="Price", ascending=True)
    
    fig = dash_table.DataTable(columns = [{"name": i, "id": i} for i in temp_df.columns],
                     data = temp_df.to_dict("records"),
                     style_cell = {"textAlign" : "left", "maxWidth": "180px", "minWidth": "90px"},
                     style_header={"backgroundColor":"#5D6D7E"},
                     style_data = {"backgroundColor": "rgba(41,47,53,1)"},
                     style_table = {"height": '500px', "overflowY":"auto"},
                     fixed_rows = {"headers": True})
    return fig

################ Update the index #####################
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/housing_overview':
        return housing_layout
    elif pathname == '/general_statistics':
        return general_stats_layout
    elif pathname == '/area_niceness':
        return area_niceness_layout
    elif pathname == '/model_page':
        return model_layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here
    

if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:





# In[ ]:




