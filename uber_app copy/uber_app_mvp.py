import flask
app = flask.Flask(__name__)

# model goes here
import numpy as np
import pandas as pd
import sklearn

#MODEL GOES HERER

# df = pd.read_csv('/Users/HudsonCavanagh/Documents/titanic.csv') #training data

# # Create dummies and drop NaNs
# df['Sex'] = df['Sex'].apply(lambda x: 0 if x == 'male' else 1)
# df = df[include].dropna()
#
# X = df[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp']]
# y = df['Survived']

PREDICTOR = .75


#
# # routes go here
# @app.route('/predict', methods=['GET'])
# def predict():
#     pclass = flask.request.args['pclass']
#     sex = flask.request.args['sex']
#     age = flask.request.args['age']
#     fare = flask.request.args['fare']
#     sibsp = flask.request.args['sibsp']
#     sex = flask.request.args['sex']
#     item = [pclass, sex, age, fare, sibsp]
#     score = PREDICTOR.predict_proba(item)
#     results = {'survival chances': score[0,1], 'death chances': score[0,0]}
#     return flask.jsonify(results)

# alternate routes

@app.route('/page')
def page():
    with open("uber_page.html", 'r') as viz_file:
        return viz_file.read()
@app.route('/result', methods=['POST', 'GET'])
def result():
    '''Gets prediction using HTML form'''
    if flask.request.method == 'POST':
        inputs = flask.request.form
        #mvp input below - commented out future inputs
        hood_id = inputs['hood_id'][0]
        day_period = inputs['day_period'][0]
        # address = inputs['address'][0]
        # time_of_day = inputs['time_of_day'][0]
        item_mvp = np.array([day_period, hood_id])

        # item = np.array([address, time_of_day, fare, sibsp])
        score_mvp = PREDICTOR.predict_proba(item_mvp)
        # score = PREDICTOR.predict_proba(item)
        results_mvp = {'survival chances': score_mvp[0,1]}
        return flask.jsonify(results_mvp)


if __name__ == '__main__':
    HOST = '127.0.0.1'
    PORT = '4000'
    app.run(HOST, PORT)
