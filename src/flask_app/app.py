from collections import Counter

from flask import Flask, render_template, request
app = Flask(__name__)


def dict_to_html(d):
    return '<br>'.join('{0}: {1}'.format(k, d[k]) for k in sorted(d))

# Home page
@app.route('/')
def index():
    return render_template('jumbotron.html', title='Predict an Airbnb Rating')

# My sentiment predictor app
@app.route('/prediction', methods=['POST'])
def show_pred():
    text = str(request.form['user_input'])
    return text


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)