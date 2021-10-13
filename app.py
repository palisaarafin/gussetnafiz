import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from numpy import inf


app = Flask(__name__,static_url_path = "/templates", static_folder = "templates")
lr_model = pickle.load(open('lr.pkl','rb'))
ridge_model = pickle.load(open('rr.pkl','rb'))
lasso_model = pickle.load(open('lasso.pkl','rb'))
dt_model = pickle.load(open('dt.pkl','rb'))
rf_model = pickle.load(open('rf.pkl', 'rb'))
knn_model = pickle.load(open('knn.pkl','rb'))
svr_model = pickle.load(open('svr.pkl','rb'))
xg_model = pickle.load(open('ab.pkl','rb'))
ab_model = pickle.load(open('ab.pkl', 'rb')) 
cb_model = pickle.load(open('cb.pkl','rb'))



scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')




@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    int_features = [float(x) for x in request.form.values()]
    print(int_features)
    
    #print(int_features)
    t=int_features[0]
    l=int_features[1]
    lc = int_features[2]
    fy = int_features[3]
    c = int_features[4]
    s = int_features[5]
    #print(e1_do,e2_do,fu_fy,type_c)
    #print(int_features[5])
    final_features=[]
    
    #final_features=final_features+[np.log(e2_do)]
    #final_features=final_features+[np.log(fu_fy)]
    
    #e1/do	e2/do	p1/do	p2/do	Nr	fu/fy
    
    if t!=0:
        final_features=final_features+[np.log(t)]
    else:
        final_features=final_features+[(t)]
        
        
    if l!=0:
        final_features=final_features+[np.log(l)]
    else:
        final_features=final_features+[(l)]
        
        
    if lc!=0:
        final_features=final_features+[np.log(lc)]
    else:
        final_features=final_features+[(lc)]
        
    if fy!=0:
        final_features=final_features+[np.log(fy)]
    else:
        final_features=final_features+[(fy)]

    if c!=0:
        final_features=final_features+[np.log(c)]
    else:
        final_features=final_features+[(c)]
        
        
    if s!=0:
        final_features=final_features+[np.log(s)]
    else:
        final_features=final_features+[(s)]
    
    final_features = [np.array(final_features)]
    print(final_features)
    

    #final_features=np.log(final_features)

    #print(final_features)
    #final_features=scaler.transform(final_features)
    
    
    print(final_features)
    
    
    rf_prediction = rf_model.predict(final_features)
    ab_prediction = ab_model.predict(final_features)
    cb_prediction = cb_model.predict(final_features)
    dt_prediction=dt_model.predict(final_features)
    knn_prediction= knn_model.predict(final_features)
    lasso_prediction = lasso_model.predict(final_features)
    lr_prediction = lr_model.predict(final_features)
    ridge_prediction = ridge_model.predict(final_features)
    svr_prediction = svr_model.predict(final_features)
    xg_prediction = xg_model.predict(final_features)
    
    

    rf_prediction_o = round(rf_prediction[0], 3)
    ab_prediction_o = round(ab_prediction[0], 3)
    cb_prediction_o = round(cb_prediction[0], 3)
    dt_prediction_o = round(dt_prediction[0], 3)
    knn_prediction_o = round(knn_prediction[0], 3)
    lasso_prediction_o=round(lasso_prediction[0], 3)
    lr_prediction_o = round(lr_prediction[0], 3)
    ridge_prediction_o = round(ridge_prediction[0], 3)
    svr_prediction_o = round(svr_prediction[0], 3)
    xg_prediction_o = round(xg_prediction[0], 3)
    
    rf_prediction_o=format(rf_prediction_o,'.3f')
    ab_prediction_o=format(ab_prediction_o,'.3f')
    cb_prediction_o=format(cb_prediction_o, '.3f')
    dt_prediction_o=format(dt_prediction_o, '.3f')
    knn_prediction_o=format(knn_prediction_o, '.3f')
    lasso_prediction_o=format(lasso_prediction_o, '.3f')
    lr_prediction_o=format(lr_prediction_o, '.3f')
    ridge_prediction_o=format(ridge_prediction_o, '.3f')
    svr_prediction_o=format(svr_prediction_o, '.3f')
    xg_prediction_o=format(xg_prediction_o, '.3f')
    
    print(rf_prediction_o,ab_prediction_o,cb_prediction_o,dt_prediction_o,knn_prediction_o,lasso_prediction_o,lr_prediction_o,ridge_prediction_o,svr_prediction_o,xg_prediction_o)
    

    
    return render_template('index.html', rf='{}'.format(rf_prediction_o), ab='{}'.format(ab_prediction_o),cb='{}'.format(cb_prediction_o), dt='{}'.format(dt_prediction_o), knn='{}'.format(knn_prediction_o), lasso='{}'.format(lasso_prediction_o), lr='{}'.format(lr_prediction_o), ridge='{}'.format(ridge_prediction_o),svr='{}'.format(svr_prediction_o), xg='{}'.format(xg_prediction_o)) 


if __name__ == "__main__":
    app.run(debug=True)
