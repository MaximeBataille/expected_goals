# Expected Goals Generator

This Dash application makes it possible to determine the expected goal of a shot according to several features. 

[Visit the app](https://expected-goals.herokuapp.com/)

![alt text](image.png)

## How is it built ?

Distance and angle of shot obviously have a very important impact. David Sumpter makes this clear on [Friends of Tracking](https://www.youtube.com/watch?v=310_eW0hUqQ).


It seemed interesting to introduce new variables to see their impact. 
Indeed, it seems that a shot close and in the axis of the goal is easy to convert into a goal. 
However, the presence of one or more opponents in the shooting angle or the body part used has an impact on the expected goal.



The features used in the model are :
- Shooting __angle__
- Shooting __distance__
- __Under Pressure__ : The player is under pressure during the shot
- __One Touch__ : The player shoots directly into the ball
- __Open Goal__ : it does not mean that there are no players in the shooting angle, but 
the player who shoots has an almost empty goal in front of him, or an empty goal, and that he can score very easily
- __Technique__ used to shoot (Normal, Lob, Volley, etc...)
- __Body Part__ used to shoot (Foot, Head, Other)
- __Type__ : Shot from open play, a penalty kick or a free kick
- __Number of players__ in the shooting angle of both teams

The model built is based on XGBoost. 

## Feature Impact

Feature Impact is displayed thanks to the [Shap package](https://github.com/slundberg/shap).

## Data

Data is provided by Statsbomb. They correspond to the matches played by Lionel Messi with FC Barcelona 
between seasons 2004/2005 and 2018/2019.

The comparison of both calibration curves and ROC AUC (Statsbomb Xg and model Xg) on the test dataset
proves our model has a good quality.

One of the next steps is to test this model on other datasets.

## Packages for model creation

```python
import pandas
import numpy
import math

import seaborn
import matplotlib
import shap

import sklearn
import xgboost

import json
import pickle

import os
```

## Packages for app creation

```python
import dash
import dash_core_components
import dash_html_components
import dash_table

import plotly
import matplotlib
import shap

import pandas
import numpy

import json
import pickle
```

## Usage - locally

In dockerImage/app 
```
python app.py
```
Then copy paste http://127.0.0.1:8050/ in a browser.

## Usage - Deployment with Heroku (free and easy)

https://dash.plotly.com/deployment

Take the same folder as dockerImage and replace Dockerfile by Procfile (/heroku_procfile/Procfile) to deploy the app with Heroku.

## Usage - To build a docker container and run the app with AWS, GCP, Azure

1- Install Docker.
https://docs.docker.com/get-docker/

2- Modify some lines of code in app.py. The goal is to have access to our app from outside the container.
```
app.run_server(host='0.0.0.0', port=8050, debug=True)
```

3- Build a docker image from the directory dockerImage.
```
docker build -t <image_name> .
```

4- Check the image is created.
```
docker images
```

5- Check that everything is working properly. The app is accessible at http://0.0.0.0:8050 .
```
docker run <image_name>
```

6- Then, deploy your app with a cloud service.

## Links and Sources

[To deploy a Dash app on Docker](https://towardsdatascience.com/how-to-use-docker-to-deploy-a-dashboard-app-on-aws-8df5fb322708).

[Friends of Tracking channel](https://www.youtube.com/channel/UCUBFJYcag8j2rm_9HkrrA7w).

[Statsbomb Open Data](https://github.com/statsbomb/open-data).