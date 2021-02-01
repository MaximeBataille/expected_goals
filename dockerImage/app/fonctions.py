def convertBool(x):
    if x=='Y':
        return True
    elif x=='N':
        return False

def transformCatFeatures(x):
    if x==0:
        return 'No'
    elif x==1:
        return 'Yes'

def rename_features(df_shap):
          col=list(df_shap.columns)
          val=list(df_shap.loc['real_data', :])

          new_cols=[]
          for c,v in zip(col, val):
            if type(v)==float or type(v)==int:
                new_col=str(round(v,2))+' = '+ c
            else :
                new_col=str(v)+' = '+ c
            new_cols.append(new_col)

          df_shap.columns=new_cols

          return df_shap
def map_label(x):
          if x<0:
            return '#1FA5C7'
          else:
            return '#ff3d40'
                           
def plotFeatureImportance(df, sign):

    from plotly.subplots import make_subplots

    categories=[]
    for i, row in df.iterrows():
        categories.append({'name':row['feature'], 'value':row['coef']})

    subplots = make_subplots(
        rows=len(categories),
        cols=1,
        subplot_titles=[x["name"] for x in categories],
        shared_xaxes=True,
        print_grid=False,
        vertical_spacing=(0.45 / len(categories)),
    )
    subplots['layout'].update(
        width=550,
        plot_bgcolor='#fff',
    )

    # add bars for the categories
    for k, x in enumerate(categories):
        subplots.add_trace(dict(
            type='bar',
            orientation='h',
            y=[x["name"]],
            x=[x["value"]],
            text=["{}".format(round(x["value"],2))],
            hoverinfo='text',
            textposition='auto',
            marker=dict(
                color="#7030a0",
            ),
        ), k+1, 1)

    # update the layout
    subplots['layout'].update(
        showlegend=False,
    )
    if sign=="neg":
        for x in subplots["layout"]['annotations']:
            x['x'] = 1
            x['xanchor'] = 'right'
            x['align'] = 'right'
            x['font'] = dict(
                size=12,
            )
    elif sign=="pos":
        for x in subplots["layout"]['annotations']:
            x['x'] = 0
            x['xanchor'] = 'left'
            x['align'] = 'left'
            x['font'] = dict(
                size=12,
            )

    # update the margins and size
    subplots['layout']['margin'] = {
        'l': 0,
        'r': 0,
        't': 20,
        'b': 1,
    }
    height_calc = 45 * len(categories)
    height_calc = max([height_calc, 350])
    subplots['layout']['height'] = height_calc
    subplots['layout']['width'] = height_calc-300
    
    for axis in subplots['layout']:
        if axis.startswith('yaxis') or axis.startswith('xaxis'):
            subplots['layout'][axis]['visible'] = False

    return subplots


def pointGrid():
    import numpy as np 
    import pandas as pd 

    array=np.zeros((121*87+1, 4))
    cpt=0
    for x in range(80+1):
        for y in range(87, 120+1):
            array[cpt,0]=x
            array[cpt,1]=y
            array[cpt,2]=cpt
            array[cpt,3]=0
            cpt+=1 
    return pd.DataFrame(data=array, columns=['x', 'y', 'customdata', 'fruit'])

def convertCoordinate(x, y):
    new_x=y
    new_y=x
    return new_x, new_y

def shotDistance(x, middle_pitch):
    import numpy as np 
    
    return np.sqrt( (middle_pitch[0]-x[0])**2  +  (middle_pitch[1]-x[1])**2)

def shotAngle(x):
    import math 
    
    near_post_coor=(120,36)
    far_post_coor=(120,44)
    near_post_dist=shotDistance(x, near_post_coor)
    far_post_dist=shotDistance(x, far_post_coor)
    
    res=(near_post_dist**2 + far_post_dist**2 - (44-36)**2) / (2*near_post_dist*far_post_dist)
    if res==1.0 or res==-1.0:
        return 0
    elif res<-1.0 or res>1.0:
        return 0
    else:
        return math.acos(res)

def plotPitch():

    import plotly.graph_objects as go

    #pitch
    df = pointGrid()
    fig = go.Figure()

    #tout terrain
    fig.add_shape(type="rect",
        x0=0, y0=87, x1=80, y1=120,
        line=dict(
            color="#9fa6b7",
            width=1,
        ),
        fillcolor='rgba(0,0,0,0)',
        layer="below",
    )

    #demi cercle surface réparation
    fig.add_shape(type="circle",
        xref="x", yref="y",
        x0=32, y0=98, x1=48, y1=111,
        line_color="#9fa6b7", line_width=1,
        layer="below"
    )

    #surface réparation
    fig.add_shape(type="rect",
        x0=18, y0=102, x1=62, y1=120,
        line=dict(
            color="#9fa6b7",
            width=1,
        ),
        fillcolor='#171b26',
        layer="below",
    )

    #6.50
    fig.add_shape(type="rect",
        x0=30, y0=114, x1=50, y1=120,
        line=dict(
            color="#9fa6b7",
            width=1,
        ),
        fillcolor='rgba(0,0,0,0)',
        layer="below",
    )

    #buts
    fig.add_shape(type="rect",
        x0=36, y0=120, x1=44, y1=122,
        line=dict(
            color="#9fa6b7",
            width=1,
        ),
        fillcolor='rgba(0,0,0,0)',
        layer="below",
    )


    fig.add_trace(go.Scatter(
                x=[40, 36, 44],
                y=[109, 120, 120],
                fill="toself",
                fillcolor='rgba(0,0,0,0)',
                opacity=0.5,
                line_width=0,
                mode='none',
                showlegend = False,
                hoverinfo='none'))

    #grid
    fig.add_trace(go.Scatter(x=df['x'], 
                             y=df['y'],
                             mode='markers',
                             marker_color='rgba(0,0,0,0)',
                             marker_size=2,
                             showlegend = False,
                             hoverinfo='none'))

    #add shot position (update with click)
    fig.add_trace(go.Scatter(x=[40], 
                             y=[109],
                             mode='markers',
                             marker_color='rgba(0,0,0,0)',
                             marker_size=13,
                             showlegend = False,
                             hoverinfo='none',
                             selectedpoints=[0],
                             marker_symbol='hexagram'))


    fig.update_layout(clickmode='event+select')


    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
    fig.update_layout(yaxis_range=[85,122], 
                      yaxis_visible=False, yaxis_showticklabels=False,
                      xaxis_visible=False, xaxis_showticklabels=False)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_xaxes(range=(-1,81))

    return fig 

