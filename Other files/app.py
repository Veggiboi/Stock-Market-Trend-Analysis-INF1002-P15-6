from flask import Flask, request, flash, render_template

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def flask_gui():
    if request.method == 'POST':
        x = request.form["x"]
    return render_template('main.html')

if __name__ == "__main__":
    app.run()