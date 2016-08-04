import flask
app = flask.Flask(__name__)

# model goes here
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('/Users/HudsonCavanagh/Documents/titanic.csv')
include = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Survived']

# Create dummies and drop NaNs
df['Sex'] = df['Sex'].apply(lambda x: 0 if x == 'male' else 1)
df = df[include].dropna()

X = df[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp']]
y = df['Survived']

PREDICTOR = RandomForestClassifier(n_estimators=100).fit(X, y)

def recommender(survive_prob):
    if survive_prob < .2:
        return "Return your tickets. Now. Just start swimming home."
    elif survive_prob < .4:
        return "I recommend spending your days immediately next to the life boats. Pocket any food you don't finish"
    elif survive_prob < .6:
        return "Hmmm I hear flying is very safe nowadays. Might get chilly on the boat."
    elif survive_prob < .8:
        return "If you love adventure, this is the trip for you! Its (slightly) more likely than not that you'll see your family again!"
    elif survive_prob <= 1:
        return "Great vacation idea! Just don't get too attached to the folks you meet on the boat. You'll (probably) be fine!"




# routes go here
@app.route('/predict', methods=['GET'])
def predict():
    pclass = flask.request.args['pclass']
    sex = flask.request.args['sex']
    age = flask.request.args['age']
    fare = flask.request.args['fare']
    sibsp = flask.request.args['sibsp']
    sex = flask.request.args['sex']
    item = [pclass, sex, age, fare, sibsp]
    score = PREDICTOR.predict_proba(item)
    results = {'survival chances': score[0,1], 'death chances': score[0,0]}
    return flask.jsonify(results)

# alternate routes

@app.route('/page')
def page():
    with open("page.html", 'r') as viz_file:
        return viz_file.read()
@app.route('/result', methods=['POST', 'GET'])
def result():
    '''Gets prediction using HTML form'''
    if flask.request.method == 'POST':
        inputs = flask.request.form
        pclass = inputs['pclass'][0]
        sex = inputs['sex'][0]
        age = inputs['age'][0]
        fare = inputs['fare'][0]
        sibsp = inputs['sibsp'][0]

        item = np.array([pclass, sex, age, fare, sibsp])
        score = PREDICTOR.predict_proba(item)
        course_of_action = recommender(score[0,1])
        results = {'survival chances': score[0,1], 'death chances': score[0,0], 'recommendation:':course_of_action}
        return flask.jsonify(results)



if __name__ == '__main__':
    HOST = '127.0.0.1'
    PORT = '4000'
    app.run(HOST, PORT)
