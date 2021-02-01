# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State 
import dash_table 
import mydcc

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import pickle
import json

import fonctions as f 

from plotly.graph_objs import *

import shap


#import model
model = pickle.load(open('data/model_xgb.pickle', 'rb'))
#import test dataset (used for shap)
X_test=pd.read_csv('data/X_test.csv')


#app
app = dash.Dash(
    __name__,
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1, maximum-scale=1.0, user-scalable=no",
        }
    ],
)
server = app.server

#plot pitch
fig=f.plotPitch()

#prepare plot for shap
fig_shap=go.Figure()
fig_shap.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis={'showgrid':False, 'zeroline':False, 'visible':False},
                yaxis={'showgrid':False, 'zeroline':False, 'visible':False}
            )

app.layout = html.Div(
    className="container scalable",
    children=[
        html.Div(
            id="banner",
            className="banner",
            children=[
                html.H6("Expected Goals Generator"),
            ],
        ),
        html.Div(
            id="upper-container",
            className="row",
            children=[
                html.Div(
                    id="upper-left",
                    className="six columns",
                    children=[
                        html.P(
                            className="section-title",
                            children='Expected Goal'),
                        html.H1(id='my-output',
                            style={'color': '#52AC86'}),
                        html.P(
                               className="section-title",
                               children='Click on the pitch to define the location of the shot'
                               ),
                        dcc.Graph(
                            id='basic-interactions',
                            figure=fig
                            ),
                        html.P(
                            className="section-title",
                            children='Feature Impact'),
                        dcc.Graph(
                            id='fig_shap',
                            figure=fig_shap
                            )
                        ]
                    ),
                html.Div(
                    id="geo-map-outer",
                    className="six columns",
                    children=[
                        html.P(
                            id="map-title",
                            children="Choose an item for each feature",
                        ),
                        html.Div(
                            id="geo-map-loading-outer",
                            children=[
                                html.Label('Under Pressure'),
                                dcc.Dropdown(
                                    id='under_pressure',
                                    options=[
                                        {'label': 'Yes', 'value': 'Y'},
                                        {'label': 'No', 'value': 'N'}
                                    ],
                                    value='N',
                                ),
                                html.Br(),
                                html.Label('One touch'),
                                dcc.Dropdown(
                                    id='one_touch',
                                    options=[
                                        {'label': 'Yes', 'value': 'Y'},
                                        {'label': 'No', 'value': 'N'}
                                    ],
                                    value='N',
                                ),
                                html.Br(),
                                html.Label('Open Goal'),
                                dcc.Dropdown(
                                    id='open_goal',
                                    options=[
                                        {'label': 'Yes', 'value': 'Y'},
                                        {'label': 'No', 'value': 'N'}
                                    ],
                                    value='N',
                                ),
                                html.Br(),
                                html.Label('Technique'),
                                dcc.Dropdown(
                                    id='technique',
                                    options=[
                                        {'label': 'Normal', 'value': 'Normal'},
                                        {'label': 'Half Volley', 'value': 'Half Volley'},
                                        {'label': 'Volley', 'value': 'Volley'},
                                        {'label': 'Lob', 'value': 'Lob'},
                                        {'label': 'Overhead Kick', 'value': 'Overhead Kick'},
                                        {'label': 'Diving Header', 'value': 'Diving Header'},
                                        {'label': 'Backheel', 'value': 'Backheel'}
                                    ],
                                    value='Normal',
                                ),
                                html.Br(),
                                html.Label('Body Part'),
                                dcc.Dropdown(
                                    id='body_part',
                                    options=[
                                        {'label': 'Foot', 'value': 'Foot'},
                                        {'label': 'Head', 'value': 'Head'},
                                        {'label': 'Other', 'value': 'Other'}
                                    ],
                                    value='Foot',
                                ),
                                html.Br(),
                                html.Label('Type'),
                                dcc.Dropdown(
                                    id='type',
                                    options=[
                                        {'label': 'Open Play', 'value': 'Open Play'},
                                        {'label': 'Free Kick', 'value': 'Free Kick'},
                                        {'label': 'Penalty', 'value': 'Penalty'}
                                    ],
                                    value='Open Play',
                                ),
                                html.Br(),
                                html.Label('Nb of players in the shooting angle'),
                                dcc.Slider(
                                    id='nb_players',
                                    min=0,
                                    max=10,
                                    marks={i: str(i) for i in range(0, 21+1)},
                                    value=2,
                                ),
                                html.Br(),
                                html.P(
                                    className="section-title",
                                    children='Notes'
                               ),
                                dcc.Markdown('''
                                __Distance__ in meters

                                __Angle__ in degrees

                                __Under Pressure__ : The player is under pressure during the shot

                                __One Touch__ : The player shoots directly into the ball

                                __Open Goal__ does not mean that there are no players in the shooting angle, but that the player who shoots has an almost empty goal in front of him, or an empty goal, and that he can score very easily.

                                __Technique__ used to shoot

                                __Body Part__ used to shoot

                                __Type__ : Shot from open play, a penalty kick or a free kick

                                __Nb of players in the shooting angle__ of both teams

                                __Binary variables__ : If Yes, the variable is selected. If No, the variable is not chosen but can still have an impact, positive or negative, on the expected goal.

                                __Feature Impact__ : The higher the bar, the greater the impact of the feature on the expected goal. Negative in blue. Positive in red.
                                '''
                                ),
                                html.A('Find all the documentation of this app', href='https://github.com/MaximeBataille/expected_goals', target='_blank'),
                            ],
                        ),
                    ],
                )
            ],
        ),
    ],
)


#########

@app.callback(
    Output(component_id='basic-interactions', component_property='figure'),
    [Input(component_id='basic-interactions', component_property='clickData')],
    [State('basic-interactions', 'figure')]
)
def updateFig(clickData, fig):

    #no shot diplays if no click
    if clickData is None:
        return fig 
    else :
        #store shot coordinates
        x=json.dumps(json.loads(json.dumps(clickData["points"][0]))["x"])
        y=json.dumps(json.loads(json.dumps(clickData["points"][0]))["y"])

        #display shooting point
        fig['data'][2]['x'] = [x]
        fig['data'][2]['y'] = [y]
        fig['data'][2]['selectedpoints']=[0]
        fig['data'][2]['marker']['color']='#1FA5C7'

        #display shooting angle
        fig['data'][0]['x'] = [x, 36, 44]
        fig['data'][0]['y'] = [y, 120, 120]
        fig['data'][0]['selectedpoints']=[0,1,2]
        fig['data'][0]['fillcolor']='#52AC86'

        return fig 

@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='under_pressure', component_property='value'),
    Input(component_id='one_touch', component_property='value'),
    Input(component_id='open_goal', component_property='value'),
    Input(component_id='technique', component_property='value'),
    Input(component_id='body_part', component_property='value'),
    Input(component_id='type', component_property='value'),
    Input(component_id='nb_players', component_property='value'),
    Input('basic-interactions', 'clickData'),
    )
def update_output_div(under_pressure_value, 
                      one_touch_value, open_goal_value, technique_value, 
                      body_part_value, type_value, nb_players_value, clickData):
    
    #no Xg diplays if no click
    if clickData==None:
        output_proba='-'
    else:
        #store shot coordinates
        x=json.dumps(json.loads(json.dumps(clickData["points"][0]))["x"])
        x=int(x)
        y=json.dumps(json.loads(json.dumps(clickData["points"][0]))["y"])
        y=int(y)
        #convert coordinates to match Statsbomb axis
        x,y=f.convertCoordinate(x, y)

        #calculate distance and angle
        middle_pitch=(120,40)
        shot_distance=f.shotDistance((x,y), middle_pitch)
        shot_angle=f.shotAngle((x,y))
        
        #modify features types
        bool_features=list(map(f.convertBool, [under_pressure_value, one_touch_value, 
                      open_goal_value]))
        cat_features=[technique_value, body_part_value, type_value]
        num_features=[shot_distance, shot_angle, nb_players_value]

        #prepare features to feed them to the model
        features=bool_features+cat_features+num_features
        shot = pd.DataFrame(data = features).T
        shot.columns = ['under_pressure', 'first_time', 'open_goal', 'technique', 'body_part', 'type', 'distance', 'angle', 'nb_players']

        #prediction
        proba_prediction = model.predict_proba(shot)
        output_proba = str(round(proba_prediction[0][1],2))

    return '{}'.format(output_proba)

@app.callback(
    Output(component_id='fig_shap', component_property='figure'),
    [Input(component_id='under_pressure', component_property='value'),
    Input(component_id='one_touch', component_property='value'),
    Input(component_id='open_goal', component_property='value'),
    Input(component_id='technique', component_property='value'),
    Input(component_id='body_part', component_property='value'),
    Input(component_id='type', component_property='value'),
    Input(component_id='nb_players', component_property='value'),
    Input('basic-interactions', 'clickData')],
    [State('fig_shap', 'figure')]
    )
def plotShap(under_pressure_value, 
                      one_touch_value, open_goal_value, technique_value, 
                      body_part_value, type_value, nb_players_value, clickData, fig):
    
    import math 

    #no Xg diplays if no click
    if clickData==None:
        return fig 
    else:
        #store shot coordinates
        x=json.dumps(json.loads(json.dumps(clickData["points"][0]))["x"])
        x=int(x)
        y=json.dumps(json.loads(json.dumps(clickData["points"][0]))["y"])
        y=int(y)
        #convert coordinates to match Statsbomb axis
        x,y=f.convertCoordinate(x, y)

        #calculate distance and angle
        middle_pitch=(120,40)
        shot_distance=f.shotDistance((x,y), middle_pitch)
        shot_angle=f.shotAngle((x,y))

        #modify types
        bool_features=list(map(f.convertBool, [under_pressure_value, one_touch_value, 
                      open_goal_value]))
        cat_features=[technique_value, body_part_value, type_value]
        num_features=[shot_distance, shot_angle, nb_players_value]

        #prepare features to feed them to the models
        features=bool_features+cat_features+num_features
        shot = pd.DataFrame(data = features).T
        shot.columns = ['under_pressure', 'first_time', 'open_goal', 'technique', 'body_part', 'type', 'distance', 'angle', 'nb_players']


        #PREPARE DATA TO DISPLAY SHAP

        #concat test dataframe and the shot
        X_all_df=pd.concat([X_test, shot], axis=0)

        #transform test_data+shot through preprocessor (OneHotEncoder and StandardScaler)
        pre_process=model.best_estimator_['preprocessor']
        X_test_pre_process=pre_process.transform(X_all_df)

        best_model=model.best_estimator_['classifier']

        #catch names of new cat features (after OneHotEncoding)
        num_features = ['distance', 'angle', 'nb_players']
        cat_features = ['under_pressure', 'first_time', 'open_goal', 
                        'technique', 'body_part', 'type']
        feature_names=num_features+list(pre_process.transformers_[1][1].steps[0][1].get_feature_names(cat_features))
        #new dataframe with preprocessed data and correct columns names
        X_test_pre_process=pd.DataFrame(data=X_test_pre_process, columns=feature_names)

        #build a dictionnary to give new names to columns
        cat_features=list(pre_process.transformers_[1][1].steps[0][1].get_feature_names(cat_features))
        new_cat_features=['Under pressure',
                            'First time',
                            'Open goal',
                            'Technique : Backheel',
                            'Technique : Diving header',
                            'Technique : Half Volley',
                            'Technique : Lob',
                            'Technique : Normal',
                            'Technique : Overhead Kick',
                            'Technique : Volley',
                            'Body part : Foot', 
                            'Body part : Head', 
                            'Body part: Other',
                            'Situation : Free Kick', 
                            'Situation : Open Play',
                            'Situation : Penalty']
        dic_conversion = {}
        for cat, new_cat in zip(cat_features, new_cat_features):
            dic_conversion[cat]=new_cat
        dic_conversion['nb_players']='Nb of players in the shooting angle'
        dic_conversion['angle']='Angle'
        dic_conversion['distance']='Distance'
        #modify columns names
        X_test_pre_process_col_renamed=X_test_pre_process.rename(columns=dic_conversion)
        X_test_pre_process_col_renamed_df=pd.DataFrame(X_test_pre_process_col_renamed)


        #shap values
        explainer = shap.TreeExplainer(best_model, X_test_pre_process_col_renamed_df)
        shap_values = explainer(X_test_pre_process_col_renamed_df)

        col=['Distance', 'Angle', 'Nb of players in the shooting angle',
           'Under pressure', 'First time', 'Open goal', 'Technique : Backheel',
           'Technique : Diving header', 'Technique : Half Volley',
           'Technique : Lob', 'Technique : Normal', 'Technique : Overhead Kick',
           'Technique : Volley', 'Body part : Foot', 'Body part : Head',
           'Body part: Other', 'Situation : Free Kick', 'Situation : Open Play',
           'Situation : Penalty']
        val=shap_values[-1].values

        #convert radian in degree, yards in meters on test_data + shot
        X_all_df['Angle']=X_all_df['angle'].apply(lambda x: round(x*(180/math.pi),1))
        X_all_df['Distance']=X_all_df['distance'].apply(lambda x: round(x/1.09361,1))

        #data of the shot
        real_data_conti=list(X_all_df[['Distance', 'Angle', 'nb_players']].iloc[-1,:])
        real_data_cat=list(X_test_pre_process[cat_features].iloc[-1,:])

        #Convert cat features value (0-->'No' and 1-->'Yes')
        real_data_cat_transformed=list(map(f.transformCatFeatures, real_data_cat))
        real_data=real_data_conti+real_data_cat_transformed

        #put shap values on a dataframe
        X=pd.DataFrame(data=np.array([val]), columns=col).T

        #put shap values, shot data and absolute value of shap_values 
        #in the same dataframe
        X['abs']=abs(X.iloc[:,0]) 
        X['real_data']=list(real_data)
        #sort dataframe by absolute values of shape values to
        #catch features having a big impact
        X=X.sort_values('abs', ascending=True)

        #10 impacting features to display
        X_to_display=X.iloc[-10:, :]
        value_others=sum(X.iloc[:-10, :].iloc[:, 0]) #sum of shap values of others features

        #drop and tranpose is easier for the following steps
        X_to_display=X_to_display.drop(['abs'], axis=1)
        X_to_display=X_to_display.T

        #rename features to have : value of the feature + feature name on Y axis
        new_X_to_display=f.rename_features(X_to_display)

        #define colors of each bar. blue if shape value is negative, red otherwise
        colors = list(map(f.map_label, list(new_X_to_display.values[0])))

        #Build the plot
        fig=go.Figure()
        fig.add_trace(go.Bar(
            x=list(X_to_display.values[0]),
            y=list(X_to_display.columns),
            marker={'color':colors,
                    'line':dict(width=0)},
            orientation='h',
            hoverinfo='none',
            width=0.5))

        #no color to be transparent
        fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

        fig.update_layout(
            width=650, 
            height=400,
            xaxis={'showgrid':False, 'zeroline':False},
            yaxis={'showgrid':True, 'zeroline' : False, 'color': "#9fa6b7" , 'gridwidth':0.01, 'gridcolor':"#3a3f4b", 'tickfont':dict(size = 14)},
                )

        return fig


if __name__ == '__main__':
    app.run_server(debug=True)
