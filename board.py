
import joblib
import streamlit as st
import pandas as pd
import warnings
import matplotlib.image as mpimg
import numpy as np
import time
import io
pd.set_option('display.max_columns',50)
warnings.filterwarnings("ignore")


global ode, scaler, km, cluster_df
ode        = joblib.load('ode.enc')
scaler     = joblib.load('scaler.sav')
km         = joblib.load("km_model.sav")
cluster_df = pd.read_csv('cluster_df.csv')

st.set_page_config(
    page_title="Style Recommender System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.image('logo.png')
st.title("Style Recommender System")



@st.cache
def load_data():

    df = pd.read_csv('case.csv', sep=',')
    df = df.drop(['CUST_ID'], axis=1).drop_duplicates(ignore_index=True)
    df.ProductGroupDescription.fillna('knitted', inplace=True)
    avg_discount = round(((df.price_before_discount-df.price_after_discount)/df.price_before_discount).mean(),2)
    print(f'Mean discount on products : {avg_discount}')
    df['price_before_discount']  = df.apply(lambda x: x.price_after_discount/(1-avg_discount) if np.isnan(x.price_before_discount) else x.price_before_discount, axis=1)

    return df

@st.cache
def transform_dataset(df):
    X           = df.drop(['CUST_AGE','price_before_discount','price_after_discount'], axis=1)

    #transforming categorical variables into numerical values
    
    transformed = ode.fit_transform(X)
    ode_df      = pd.DataFrame(transformed, columns = X.columns)

    #Merging transformed data 
    data        = df.drop(columns=X.columns)
    data        = pd.concat([ode_df,data], axis=1)

    #normalizing the variables using MinMaxScaler 
    
    scaled      = scaler.fit_transform(data)
    scaled_df   = pd.DataFrame(scaled, columns = data.columns)
    return scaled_df


def get_recommendations(page,df=cluster_df, n=5, cross_sell=False):  
    X           = page.drop(['CUST_AGE','price_before_discount','price_after_discount'], axis=1)

    #transforming categorical variables into numerical values
    transformed = ode.transform(X)
    ode_df      = pd.DataFrame(transformed, columns = X.columns)
    
    #Merging transformed data 
    data       = page.copy().reset_index(drop=True)
    data.drop(X.columns, axis=1, inplace =True)
    data       = pd.concat([ode_df,data], axis=1)
    
    #normalizing the variables using MinMaxScaler 
    scaled     = scaler.transform(data)
    scaled_df  = pd.DataFrame(scaled, columns = data.columns)
      
    cluster_map = {0:12, 12:0, 3:8, 8:3, 2:5,  5:2, 1:6, 6:1, 5:11, 11:5, 4:11, 7:11, 9:11, 10:6}

    cluster    = km.predict(scaled_df)[0]
    if not cross_sell:
        return df.iloc[np.random.choice(df[df.label==cluster].index,n),:], cluster
    else:
        return df.iloc[np.random.choice(df[df.label==cluster_map[cluster]].index,n),:], cluster, cluster_map[cluster]

data_load_state = st.text('Loading data...')
df = load_data()

data_load_state.text('Transforming data. . ')
scaled_df = transform_dataset(df)
sum_stat  = scaled_df.describe()

elb = mpimg.imread('elbow.png')
img = mpimg.imread('clusters.png')


data_load_state.text('Everything loaded!')
time.sleep(0.5)
data_load_state.text('')

with st.sidebar:
    add_radio = st.radio("Select Pipeline Stage",['Cleaned Data','Transformed Data','Groups in Data','Elbow Plot','Clusters','Get Recommendations','Cross-sell'])

if add_radio=='Cleaned Data':
    st.subheader('Cleaned Data')
    st.write(df)
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.subheader("Overview of fields in cleaned dataset")
    st.text(s)

elif add_radio == 'Transformed Data':
    st.subheader('Transformed Data')
    st.write(scaled_df)
    st.subheader("Summary stats of fields in transformed dataset")
    st.write(sum_stat)

elif add_radio == 'Groups in Data':
    st.subheader('Developing intuition for the groups in the data')
    groups = df.groupby('TargetGroupDescription')['ProductClusterDescription'].unique()
    st.write(groups)
    
    col1, col2 = st.columns(2)
    col1.subheader("Women")
    col2.subheader("Men")
    col1.write(groups.loc['Women'])
    col2.write(groups.loc['Men'])


elif add_radio == 'Elbow Plot':
    
    st.subheader("Sum of square distances for different values for number of clusters")
    st.image(elb)

elif add_radio == 'Clusters':
    
    st.subheader("Visualizing clusters composed using two the principal components")
    st.image(img)

elif add_radio == 'Get Recommendations':
    
    st.subheader("Get style recommendation")
    ix   = st.number_input('Enter row number to get recommendation', min_value=0, max_value=503823, value=0)
    n    = st.number_input('Number of recommendations', min_value=1, max_value=10, value=5)
       
    page   = pd.DataFrame(df.iloc[int(ix),:]).transpose()
    rec_df,cluster = get_recommendations(page=page, n=int(n))
    
    st.text(f"Selected Page cluster :{cluster}")
    st.write(df[df.index==int(ix)])
    st.text(f"Recommedated pages from cluster {cluster}:")
    st.dataframe(rec_df)
    
elif add_radio == 'Cross-sell':
    
    st.subheader("Get cross-sell recommendations")
    ix   = st.number_input('Enter row number to get recommendation', min_value=0, max_value=503823, value=0)
    n    = st.number_input('Number of recommendations', min_value=1, max_value=10, value=5)

    page   = pd.DataFrame(df.iloc[int(ix),:]).transpose()  
    rec_df,cluster, cluster2 = get_recommendations(page=page, n=int(n), cross_sell=True)
    
    st.text(f"Selected Page cluster :{cluster}")
    st.write(df[df.index==int(ix)])
    st.text(f"Cross-sell recommendation from cluster {cluster2}")
    st.write(rec_df)    

