from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Route for aim optimizer page
@app.route("/aim_optimizer")
def aim_optimizer():
    return render_template("aim_optimizer.html")

# Route for distribution calculator page
@app.route("/distribution")
def distribution():
    return render_template("distribution.html")

# Route for check-out strategy page
@app.route("/check_out")
def check_out():
    return render_template("check_out.html")

# Route for score tracker page
@app.route("/score_tracker")
def score_tracker():
    return render_template("score_tracker.html")

if __name__ == "__main__":
    app.run(debug=True)