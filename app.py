from flask import Flask, render_template, request, session
import pickle
import numpy as np

# model = pickle.load(open('iri.pkl', 'rb'))
model = pickle.load(open('iri.pkl','rb'))
model_two = pickle.load(open('failure.pkl','rb'))

app = Flask(__name__)
app.secret_key="itssecret1234"



@app.route('/')
def man():
    return render_template('index.html')

@app.route('/pred_page')
def pred_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data9 = request.form['i']
    data10 = request.form['j']
    data11 = request.form['k']
    data12 = request.form['l']
    data13 = request.form['m']
    session['age'] = data1
    session['gender'] = data2
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13]],dtype=object)
    pred = model.predict(arr)
    # print(pred[0])
    # return "hii"
    report = {}
    # Example: Checking resting blood pressure (trestbps)
    # trestbps_normal_max = 120
    # if int(data4) > trestbps_normal_max:
    #     report["Resting Blood Pressure"] = f"{data4} mm Hg (Normal: <120/80 mm Hg)."
    if pred==0:
        return render_template('pred_z.html')
    elif pred==1:
        return render_template('pred_o.html',report=report)
    elif pred==2:
        return render_template('pred_t.html')
    elif pred==3:
        return render_template('pred_th.html')
    elif pred==4:
        return render_template('pred_f.html')
    

@app.route('/pred_fail_form')
def pred_fail():
    age = session.get('age', '')
    gender = session.get('gender', '')
    return render_template('pred_fail.html',age=age, gender=gender)


@app.route('/pred_failure', methods=['POST'])
def pred_failure():
    sec_data1 = request.form['a1']
    sec_data2 = request.form['b1']
    sec_data3 = request.form['c1']
    sec_data4 = request.form['d1']
    sec_data5 = request.form['e1']
    sec_data6 = request.form['f1']
    sec_data7 = request.form['g1']
    sec_data8 = request.form['h1']
    sec_data9 = request.form['i1']
    sec_data10 = request.form['j1']
    sec_data11 = request.form['k1']
    # sec_data12 = request.form['l1']
    arr = np.array([[sec_data1, sec_data2, sec_data3, sec_data4, sec_data5, sec_data6, sec_data7, sec_data8, sec_data9, sec_data10, sec_data11]],dtype=object)
    pred_sec = model_two.predict(arr)
    # print(pred[0])
    # return "hii"
    if pred_sec==0:
        return render_template('false.html')
    elif pred_sec==1:
        return render_template('true.html')


if __name__ == "__main__":
    app.run(debug=True)