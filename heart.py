import streamlit as st
import pickle
# from joblib import load
import datetime as dt
# import skops.io as sio
from sklearn.metrics import f1_score,r2_score


st.title("Heart Disease Prediction")
now=dt.datetime.today()
today=now.strftime("%d.%m.%Y")
st.header(f"heute ist der {today}")

def prediction():
    global result,result_text
    with open("model.pkl","rb") as f:
        clf=pickle.load(f)
    try:
        print(clf)
    except Exception as e:
        print(e)
    result=clf.predict([[40,1,250,0,180,180]])
    print(result)
    # print(r2_score(result,result))
    if result==1:
        result_text="Du hast mit ca. 80% wahrscheinlich eine Herzerkrankung"
        
    else:
        result_text="Du hast mit ca .80% wahrscheinlichkeit keine Herzerkrankung"
    return result,result_text        
prediction()    


st.text(result_text)





