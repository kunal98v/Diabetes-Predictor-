import pandas as pd
import numpy as np 
from sklearn.linear_model import LogisticRegression
from flask import Flask,render_template,request

app = Flask(__name__)


df = pd.read_csv('C:/Users/Kunal/Desktop/DIAB/diabetes.csv')

def split(data,ratio):
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


train , test = split(df,0.2)

X_train = train[['Pregnancies','Glucose','BP','Insulin','BMI','Age']].to_numpy()
X_test =  test[['Pregnancies','Glucose','BP','Insulin','BMI','Age']].to_numpy()

Y_train= train[['Outcome']].to_numpy().reshape(615,)
Y_test= test[['Outcome']].to_numpy().reshape(153,)

clf = LogisticRegression()
clf.fit(X_train,Y_train)

@app.route('/',methods=['POST','GET'])
def index():
    if request.method=='POST':
        dic =request.form
        preg=int(dic['preg'])
        glu=int(dic['glu'])
        bp=int(dic['bp'])
        insu=int(dic['insu'])
        bmi=int(dic['bmi'])
        age=int(dic['age'])

        inf = (clf.predict_proba([[preg,glu,bp,insu,bmi,age]])[0][1])
        kun= round(inf,2)*100
        print(kun)
        return render_template('result.html',result=kun)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)