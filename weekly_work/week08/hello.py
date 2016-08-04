import flask
app = flask.Flask(__name__)

@app.route("/")
def hello():
    return "<h2> Hello World! </h2>"

@app.route("/greet/<name>")
def greet(name):
    '''Say hello to your first parameter'''
    return "Hello, %s!" %name

if __name__ == '__main__':
    HOST = '127.0.0.1'
    app.run(HOST)
