from flask import Flask, request, flash, render_template

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def flask_gui():
    
    if request.method == 'POST':
        ticker = request.form.get('ticker')
        duration = request.form.get('duration')
        sma = request.form.get('sma')

        print(ticker)
        print(duration)
        print(sma)

    return render_template('main.html')

if __name__ == "__main__":
    app.run()