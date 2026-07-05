import Model
# pyrefly: ignore [missing-import]
from flask import Flask, render_template, request

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/'
model = Model.MyModel()

@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files['image']
        file.save('./static/image.png')
        (name, prob) = model.predict()
        print(name, prob)
        if name == 'NORMAL':
            confidence = (1 - prob) * 100
            prob = f'The model is {confidence:.2f}% confident that this X-ray is Normal (No Pneumonia detected).'
        elif name == 'PNEUMONIA':
            confidence = prob * 100
            prob = f'The model is {confidence:.2f}% confident that this X-ray shows signs of Pneumonia.'

        return render_template("results.html", predicted_name=name, probability=prob)


app.run(debug=True)
