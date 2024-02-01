import streamlit as st
import numpy as np
import pickle
import pandas as pd
#from pathlib import Path
import datetime as dt
import plotly.express as px
from datetime import date
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib


st.set_page_config(layout="wide")
st.markdown("""# Nomin United Hypermarket - 10/13/2023""")
st.markdown("""Энэхүү суурь загвар нь Номин Юнайтед ХМ-ын SKU-үүд дээр суурилсан ба :blue["2022-01-01"]-ээс :blue["2023-09-01"] хүртэлх борлуулалт дээр сургагдсан. 
Одоогоор ашиглагдаж байгаа аргачлалууд нь: 
- Тухайн салбарт өндөр борлуулалттай (тоогоор) 50 бараа тус бүр
- Барааны ангилал
- Барааны үнэ
- Барааны брэнд
- Барааны түгээгч
- Борлуулалтын өдрөөр бүлэглэлт
- :red[Үлдэгдэлгүй (дутагдалтай) өдрүүдийн борлуулалтыг оруулаагүй]
- Он цаг, улиралын шинж чанарууд (7 хоногийн өдөр, хагас бүтэн сайн, амралтын өдрүүд, сар, жил)
- Хямдралтай өдрүүд
- 7 ба 14 хоногийн өмнөх борлуулалтыг оруулсан (lags)""")
inv = pd.read_csv("RemainingInventory.csv") #inventory information
df = pd.read_csv("Readable.csv")            #item information (non-grouped, non-dummy)
data = pd.read_csv("CompleteData.csv")      #item information (non-grouped, non-dummy)
data = data.drop(columns=['Unnamed: 0'])
items = pd.read_csv("ItemList.csv")
items['itemkey'] = items['itemkey'].astype('str')
category_list = items['category'].unique()
category_choice = st.selectbox("Ангилал сонгох", category_list)
item_list = items[items['category']==category_choice]['item'].unique()
item_choices = st.multiselect('Бараа сонгох',item_list)
#################################################################################################################
X_test = data[(data['date'] >= '2023-08-01') & (data['date'] < '2023-09-01')].drop(columns=['date','quantity','itemkey'])
X_test_ik = data[(data['date'] >= '2023-08-01') & (data['date'] < '2023-09-01')]['itemkey'].astype('str')
y_test_date = data[(data['date'] >= '2023-08-01') & (data['date'] < '2023-09-01')]['date']
y_test = data[(data['date'] >= '2023-08-01') & (data['date'] < '2023-09-01')]['quantity']
X_valid = data[(data['date'] >= '2023-09-01')].drop(columns=['date','quantity','itemkey'])
y_valid = data[(data['date'] >= '2023-09-01')]['quantity']
X_valid_ik = data[(data['date'] >= '2023-09-01')]['itemkey'].astype('str')
y_valid_date = data[(data['date'] >= '2023-09-01')]['date']
#with open("simple_rfr_model.pkl", 'rb') as pickle_file:
#    model = pickle.load(pickle_file)
with open("simple_rfr_model.pkl", 'rb') as model_file:
    model = joblib.load(model_file)
test_predictions = model.predict(X_test)
val_predictions = model.predict(X_valid)
X_t = X_test.copy()
X_t['label'] = y_test
X_t['preds'] = test_predictions
X_v = X_valid.copy()
X_v['label'] = y_valid
X_v['preds'] = val_predictions
X_t['dates'] = y_test_date
X_v['dates'] = y_valid_date
X_v['item_names'] = df[(df['Date'] >= '2023-09-01')]['item']
X_t['item_names'] = df[(df['Date'] >= '2023-08-01') & (df['Date'] < '2023-09-01')]['item']
#################################################################################################################
temporary = X_v[X_v['item_names'].isin(item_choices)].set_index('dates')[['item_names','label','preds']]         
col1, col2, col3 , col4, col5, col6, col7 = st.columns(7)
temp_xv = X_v[['dates','item_names','base_price','price','preds','label']].copy()
temp_xv = temp_xv.rename(columns={'dates':'Огноо','item_names':'SKU','base_price':'Үндсэн үнэ','price':'Борлогдсон үнэ','preds':'Таамаглал','label':'Бодит'})

with col1:
    pass
with col2:
    pass
with col3:
    pass
with col5:
    pass
with col6:
    pass
with col4:
    x = st.button("Унших")
if x:
    for i in item_choices:
        with st.container():
            col_x,col_y,col_z=st.columns(3)
            with col_y:
                st.markdown(f"#### {i}")
            col_a,col_b=st.columns(2)
            with col_a:
                st.markdown("#### Хүснэгт")
                st.write(temp_xv[temp_xv['SKU']==i].reset_index(drop=True))
                lbl = temp_xv[temp_xv['SKU']==i]['Бодит'].astype('float')
                prd = temp_xv[temp_xv['SKU']==i]['Таамаглал'].astype('float')
                rmse = mean_squared_error(lbl,prd, squared=False)
                r2 = r2_score(lbl, prd)
                mape = mean_absolute_percentage_error(lbl,prd)
                errors = pd.DataFrame()
                st.markdown(f"RMSE: {rmse}")
                st.markdown(f"MAPE: {mape}")
                st.markdown(f"R2: {r2}")
            with col_b:
                st.markdown("#### Граф")
                fig = px.line(temp_xv[temp_xv['SKU']==i].set_index('Огноо')[['Бодит','Таамаглал']])
                fig.update_layout(
                    xaxis_title="Огноо",
                    yaxis_title="Борлуулалт")
                st.plotly_chart(fig)

