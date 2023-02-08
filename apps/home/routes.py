# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
import pandas as pd
from multiprocessing import connection
from apps.home import blueprint
from flask import render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound
# from sklearn.externals import joblib
# 필요한 파일 import
import time


import numpy as np
import matplotlib.pyplot as plt  #그래프 만들기위해
import seaborn as sns
import glob
import json
import pickle


from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import select
from sqlalchemy import or_
from sqlite3 import Cursor
from pycaret.regression import *

#모델 가져오기
# model = pickle.load(open('C:/dev/sale/flask-pixel/apps/static/assets/huber_last.pkl', 'rb'))

model = load_model('C:/dev/sale/flask-pixel/apps/static/assets/huber_last')
df_store = pd.read_csv('C:/dev/sale/flask-pixel/apps/static/assets/store.csv')
def model1(a,b):
    
    #데이터프레임 가져오기
    
    df_store = pd.read_csv('C:/dev/sale/flask-pixel/apps/static/assets/store.csv')
    df_dong = pd.read_csv('C:/dev/sale/flask-pixel/apps/static/assets/dong.csv')
    #매핑시킬 데이터 가져오기

    input_sclas = df_store[df_store['induty_sclas_nm']==a]['induty_sclas_code'].values[0]
    input_dong = df_dong[df_dong['legaldong_nm']==b]['legaldong_code'].values[0]

    return input_sclas,input_dong

    # 모델 피처 매핑하기
def model_dt_func(input_l, input_i):

    main_dt_6 = pd.read_csv('C:/dev/sale/flask-pixel/apps/static/assets/main_dt_4.csv')
    model_dt = main_dt_6[(main_dt_6['induty_sclas_code'] == input_l)&(main_dt_6['legaldong_code']==input_i)]
    model_dt = model_dt.drop(['induty_sclas_code','legaldong_code'],axis=1)
    return model_dt


# 지역 mysql에서 들고오기
def regionList():  
    db_uri = 'mysql+mysqldb://root:yejin2138@localhost:3306/graphdb'
    engine = create_engine(db_uri)
    conn = engine.connect()

    result1=engine.execute('SELECT * FROM region_table')
    dong=result1.fetchall()

    return dong
    
# 중분류 mysql에서 들고오기   s
def mstoreList():  
    db_uri = 'mysql+mysqldb://root:yejin2138@localhost:3306/graphdb'
    engine = create_engine(db_uri)
    conn = engine.connect()

    result2=engine.execute('SELECT * FROM store_table')
    mstore=result2.fetchall()
    return mstore

# 소분류 mysql에서 들고오기   s
def sstoreList():  
    db_uri = 'mysql+mysqldb://root:yejin2138@localhost:3306/graphdb'
    engine = create_engine(db_uri)
    conn = engine.connect()

    result3=engine.execute('SELECT * FROM small_store')
    sstore=result3.fetchall()
    return sstore

def graph1(a):

    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] =False

    df_1=pd.read_csv("C:/dev/sale/flask-pixel/apps/static/assets/sales.csv")

    value_1 = df_1[(df_1['legaldong_nm']== a )]

    data_gr = value_1["induty_mlsfc_nm"].value_counts() # 업종별 이용 횟수
    x = data_gr.index # 데이터 목록 넣기
    labels = [str(i) for i in x]

    values = data_gr.values # 데이터 값 넣기


    ## 데이터 준비
    colors = ['#ff9999','#ffc000', '#d395d0', 'yellow', '#8fd9b6','cornflowerblue','yellowgreen']# 색상
    frequency = values ## 데이터 값

    ## 데이터 라벨, 빈도수, 색상을 빈도수를 기준으로 정렬해야한다.
    labels_frequency = zip(labels,frequency,colors) 
    labels_frequency = sorted(labels_frequency,key=lambda x: x[1],reverse=True)
    
    sorted_labels = [x[0] for x in labels_frequency] ## 정렬된 라벨
    sorted_frequency = [x[1] for x in labels_frequency] ## 정렬된 빈도수
    sorted_colors = [x[2] for x in labels_frequency] ## 정렬된 색상
    
    fig = plt.figure(figsize=(10,10)) ## 캔버스 생성
    fig.set_facecolor('white') ## 캔버스 배경색을 하얀색으로 설정
    ax = fig.add_subplot() ## 프레임 생성
    
    pie = ax.pie(sorted_frequency, ## 파이차트 출력
        startangle=90, ## 시작점을 90도(degree)로 지정
        counterclock=False, ## 시계방향으로 그려짐
        colors = sorted_colors, ## 색상 지정
        wedgeprops = {'width':0.7} ##도넛모양만들기
        )
    
    total = np.sum(frequency) ## 빈도수 합
    
    threshold = 5
    sum_pct = 0 ## 퍼센티지
    count_less_5pct = 0 ## 5%보다 작은 라벨의 개수
    spacing = 0.1
    for i,l in enumerate(sorted_labels):
        ang1, ang2 = ax.patches[i].theta1, ax.patches[i].theta2 ## 파이의 시작 각도와 끝 각도
        center, r = ax.patches[i].center, ax.patches[i].r ## 파이의 중심 좌표
        
        ## 비율 상한선보다 작은 것들은 계단형태로 만든다.
        if sorted_frequency[i]/total*100 < threshold:
            x = (r/2+spacing*count_less_5pct)*np.cos(np.pi/180*((ang1+ang2)/2)) + center[0] ## 텍스트 x좌표
            y = (r/2+spacing*count_less_5pct)*np.sin(np.pi/180*((ang1+ang2)/2)) + center[1] ## 텍스트 y좌표
            count_less_5pct += 1
        else:
            x = (r/2)*np.cos(np.pi/180*((ang1+ang2)/2)) + center[0] ## 텍스트 x좌표
            y = (r/2)*np.sin(np.pi/180*((ang1+ang2)/2)) + center[1] ## 텍스트 y좌표
        
        ## 퍼센티지 출력
        if i < len(labels) - 1:
            sum_pct += float(f'{sorted_frequency[i]/total*100:.2f}')
            ax.text(x,y,f'{sorted_frequency[i]/total*100:.2f}%',ha='center',va='center',fontsize=18)
        else: ## 마지막 파이 조각은 퍼센티지의 합이 100이 되도록 비율을 조절
            ax.text(x,y,f'{100-sum_pct:.2f}%',ha='center',va='center',fontsize=18)
    
    plt.legend(pie[0],sorted_labels) ## 범례
    plt.title(a + ' 업종별 이용비율', fontsize=18) #제목
    plt.savefig("C:/dev/sale/flask-pixel/apps/static/assets/"+a+"out1.png")

    return ("static/assets/"+a+"out1.png")

def graph2(a):
    
    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] =False

    df_1=pd.read_csv("C:/dev/sale/flask-pixel/apps/static/assets/sales.csv")

    value_1 = df_1[(df_1['legaldong_nm']== a )]

    data_gr1 = value_1.groupby('induty_mlsfc_nm')['sales'].sum() #업종별 매출액
    data_gr1=data_gr1.sort_values( ascending=True).head(7)
    x1 = data_gr1.index  
    labels = [str(i) for i in x1]
    values = data_gr1.values # 데이터 값 넣기

    list_x = x1 # 데이터 인덱스
    list_y = values #데이터 값
    plt.figure(figsize=(9,9))
    plt.title( a + ' 업종별 매출액', fontsize=18) #제목
    plt.xlabel('매출액(억단위)') 
    plt.barh(list_x, list_y, color = 'lightskyblue')
    plt.savefig("C:/dev/sale/flask-pixel/apps/static/assets/"+a+"out2.png")

    return ("static/assets/"+a+"out2.png")

def graph3(a):

    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] =False

    df_1=pd.read_csv("C:/dev/sale/flask-pixel/apps/static/assets/sales.csv")

    value_1 = df_1[(df_1['legaldong_nm']== a )]

    data_gr2 = value_1.groupby('induty_mlsfc_nm')['sales'].sum() #업종별 매출액
    data_gr2=data_gr2.sort_values( ascending=False).head(7)
    x1 = data_gr2.index  
    labels = [str(i) for i in x1]
    max=labels[0]

    value_1=value_1[value_1['induty_mlsfc_nm']==max]

    data_gr2 = value_1.groupby('sexdstn')['sales'].sum()  # 성별에 따른 매출액 합
    x2 = data_gr2.index   #데이터 인덱스 넣기
    labels = [str(i) for i in x2]

    values2 = data_gr2.values # 데이터 값 넣기

    explode = [0,10]

    plt.figure(figsize=(8,8))
    colors = ['lightsalmon', 'powderblue']
    plt.pie(values2, explode=explode,labels=labels ,startangle=90,shadow=True,autopct='%.1f%%', colors=colors, textprops = {'fontsize':18})
    plt.title(max+ ' 성비 ', fontsize=18) #제목
    plt.legend(labels=labels, loc='upper left', bbox_to_anchor=(1.0, 1))

    plt.savefig("C:/dev/sale/flask-pixel/apps/static/assets/"+a+"out3.png")

    return ("static/assets/"+a+"out3.png")

@blueprint.route('/index')
@login_required
def index():

    return render_template('home/index.html', segment='index')

@blueprint.route('/contents01', methods=['POST',"GET"])
@login_required
def contensts01():
    
    # 지역
    dong=regionList()
    #튜플 형태로 출력되기 때문에 튜플의 첫 원소만 출력
    dong=[x[0] for x in dong]

    # 중분류
    mstore=mstoreList()
    #튜플 형태로 출력되기 때문에 튜플의 첫 원소만 출력
    # mstore=[y[0] for y in mstore]
    mstore = df_store['induty_mlsfc_nm'].unique()

    # 소분류
    sstore=sstoreList()
    #튜플 형태로 출력되기 때문에 튜플의 첫 원소만 출력
    # sstore=[z[0] for z in sstore]
    sstore = df_store['induty_sclas_nm'].unique()

    return render_template('home/contents01.html' ,x=dong,mstore=mstore,sstore=sstore)

@blueprint.route('/contents03', methods=['POST',"GET"])
@login_required
def contensts03():
    # 지역
    dong=regionList()
    #튜플 형태로 출력되기 때문에 튜플의 첫 원소만 출력
    dong=[x[0] for x in dong]
    
    return render_template('home/contents03.html' ,x=dong)

@blueprint.route('/predict',methods=['GET','POST'])
@login_required
def predict():
    if request.method=='GET':
        sclas=request.args.get("sclas")
        town=request.args.get("town")
        
        a,b=model1(sclas,town)
        
        # input_sclas = df_store[df_store['induty_sclas_nm']==sclas]['induty_sclas_code']
        # input_dong = df_dong[df_dong['legaldong_nm']==town]['legaldong_code']
        #list = model_dt_func(a, b).values.tolist()
        # arr = np.array()
        #pred = model.predict( list )

        df = model_dt_func(a, b)
        print(type(df))
        
        pred = predict_model(model, data= df)
        pred = pred['Label'].values[0]
        return render_template('home/predict.html',sclas=sclas,town=town, pred=pred)

@blueprint.route('/graph',methods=['GET','POST'])
@login_required
def graph():
    if request.method=='GET':
        town=request.args.get("town")
        gra1=graph1(town)
        gra2=graph2(town)
        gra3=graph3(town)
        return render_template('home/graph.html',town=town,gra1=gra1,gra2=gra2,gra3=gra3)



@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None

    