from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', insurance_cost = '')

@app.route('/predict', methods=['POST'])
def predict():
    model, nama, jenis, bentuk, kategori, dosis, harga = [x for x in request.form.values()]
    data = [[nama.lower(), jenis.lower(), bentuk.lower(), kategori.lower(), dosis, harga]]
    
    model_file = open(model, 'rb')
    model = pickle.load(model_file, encoding='bytes')

    print(data)

    x_trans = OrdinalEncoder()
    X = x_trans.fit_transform(data) 
    print(X)
    X = np.reshape(X, (1, -1))
    print(X)
    prediction = model.predict(X)
    output = round(prediction[0], 1)

    return render_template('index.html', 
            insurance_cost=output, 
            nama=nama, 
            jenis=jenis,
            bentuk=bentuk,
            kategori=kategori,
            dosis=dosis,
            harga=harga,
    )

if __name__ == '__main__':
    app.run(debug=True)