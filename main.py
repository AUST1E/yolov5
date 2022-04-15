from cProfile import run
import datetime
from io import StringIO
import io
from multiprocessing import connection
from optparse import Values
from pathlib import Path
from click import option
import cv2
from cv2 import VideoCapture
from cv2 import VideoWriter
from matplotlib.pyplot import connect
import numpy as np
import streamlit as st
import time
from detect import detect_image,detect_video
import os
import sys
import argparse
from PIL import Image
import pyodbc
import altair as alt
import pandas as pd
import re
UPLOAD_FOLDER = 'data/images'

# Initialize connection.
# Uses st.experimental_singleton to only run once.
@st.experimental_singleton
def init_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};SERVER="
        + st.secrets["server"]
        + ";DATABASE="
        + st.secrets["database"]
        + ";UID="
        + st.secrets["username"]
        + ";PWD="
        + st.secrets["password"]
    )

conn = init_connection()

def reconnect(conn):
    conn.close()
    conn1 = init_connection()
    return conn1
# Perform query.
# Uses st.experimental_memo to only rerun when the query changes or after 10 min.
@st.experimental_memo(ttl=600)
def select(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()

def update(query):
    conn.cursor().execute(query)

def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)

def change(videos,path):
    ready_images = []
    import cv2

    for img in videos['Images']:
        image = cv2.imread(img.fileName)
        ready_images.append(image)

    fourcc = cv2.VideoWriter_fourcc(*'H264')

    video_name = videos['Images'][0].gifLocationPath + "//" + videos['Name']
    frame = cv2.imread(videos['Images'][0].fileName)
    height, width, layers = frame.shape

    video_name = video_name[:-4]+".mp4"
    video = cv2.VideoWriter(video_name, fourcc, 20.0, (width, height))

    for image in ready_images:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()

def change2(image,path):
    import cv2
    capture = cv2.VideoCapture(path)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)   
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    st.text(image)
    st.text(path)
    #image = cv2.imread(img.fileName)

    fourcc = cv2.VideoWriter_fourcc(*'H264')

    video_name = image
    frame = cv2.imread(path)
    #height = frame.shape[0]
    #width = frame.shape[1]

    video_name = video_name[:-4]+".mp4"
    video = cv2.VideoWriter(path, fourcc, frame_count, (1920, 1080))
    video<<image

    cv2.destroyAllWindows()
    video.release()
    
def str1date(str,date_format="%Y%m%d"):
    import re
    date = datetime.datetime.strptime(str, date_format)
    return date


def str2date(str,date_format="%Y%m%d"):
    import re
    str1 = ''.join(re.findall(r'.{4}', str)[0])
    str2 = ''.join(re.findall(r'.{2}', re.findall(r'.{5}', str)[1])[0])
    str3 = ''.join(re.findall(r'.{2}', re.findall(r'.{2}', str)[4])[0])
    date = datetime.datetime.strptime(str1 + str2 +str3, date_format)
    return date


def count_qualified(args):
    count = 0
    for row in args:
        if row[0] == False:
            count += 1
    return count

def count_num(args):
    count = 0
    for row in args:
        count += row[0]
    return count
def search():
    
    #conn = reconnect(conn)
    rows = select("select * from board_detection_results")
    Loose_fan_screws = select("SELECT Loose_fan_screws from board_detection_results")
    Missing_fan_screws = select("SELECT Missing_fan_screws from board_detection_results")
    Loose_board_screws = select("SELECT Loose_board_screws from board_detection_results")
    Board_screw_model_error = select("SELECT Board_screw_model_error from board_detection_results")
    Missing_board_screws = select("SELECT Missing_board_screws from board_detection_results")
    Incorrect_fan_wiring = select("SELECT Incorrect_fan_wiring from board_detection_results")
    Is_Qualified = select("SELECT Qualified from board_detection_results")

    num_Loose_fan_screws = count_num(Loose_fan_screws)
    num_Missing_fan_screws = count_num(Missing_fan_screws)
    num_Loose_board_screws = count_num(Loose_board_screws)
    num_Board_screw_model_error = count_num(Board_screw_model_error)
    num_Missing_board_screws = count_num(Missing_board_screws)
    num_Incorrect_fan_wiring = count_num(Incorrect_fan_wiring)
    num_unQualified = count_qualified(Is_Qualified)
    total_num =  select("SELECT COUNT(*) FROM board_detection_results")[0][0]
    num_Qulified = total_num - num_unQualified
    if (total_num != 0):Qualified_Rate = num_Qulified / total_num
    else: Qualified_Rate = 0

    detail = pd.DataFrame(
        index=range(1), 
        columns=['风扇螺丝松动         ','缺少风扇螺丝           ','主板螺丝松动          ','主板螺丝型号不正确            ',
        '缺少主板螺丝          ','风扇接线不正确          ','不合格       '])           

    detail.loc[0] = [num_Loose_fan_screws,num_Missing_fan_screws,num_Loose_board_screws,num_Board_screw_model_error,
    num_Missing_board_screws,num_Incorrect_fan_wiring,num_unQualified]

    # Print results.
    total = pd.DataFrame(
        index=range(3), 
        columns=['主板编号         ','风扇螺丝松动         ','缺少风扇螺丝         ','主板螺丝松动         ','主板螺丝型号不正确         ',
        '缺少主板螺丝         ','风扇接线不正确         ','是否合格         ','检测时间         '])              
    count = 0
    for row in rows:
        total.loc[count] = [row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8]]
        count+=1

    #柱状图
    source = pd.DataFrame({
    '缺陷种类': ['风扇螺丝松动         ','缺少风扇螺丝           ','主板螺丝松动          ','主板螺丝型号不正确            ',
        '缺少主板螺丝          ','风扇接线不正确          '],
    '数量': [num_Loose_fan_screws,num_Missing_fan_screws,num_Loose_board_screws,num_Board_screw_model_error,
    num_Missing_board_screws,num_Incorrect_fan_wiring]
    })

    bar = alt.Chart(source).mark_bar().encode(
        y='缺陷种类',
        x='数量'
    )

    
    #饼状图
    source = pd.DataFrame({"缺陷检测结果": ['风扇螺丝松动         ','缺少风扇螺丝           ','主板螺丝松动          ','主板螺丝型号不正确            ',
        '缺少主板螺丝          ','风扇接线不正确          '], "value": [num_Loose_fan_screws,num_Missing_fan_screws,num_Loose_board_screws,num_Board_screw_model_error,
    num_Missing_board_screws,num_Incorrect_fan_wiring]})

    pie_cat = alt.Chart(source).mark_arc().encode(
        theta=alt.Theta(field="value", type="quantitative"),
        color=alt.Color(field="缺陷检测结果", type="nominal"),
    )



    #合格率饼状图
    source = pd.DataFrame({"产品检测结果": ['合格','不合格'], "value": [num_Qulified,num_unQualified]})

    pie_rate = alt.Chart(source).mark_arc().encode(
        theta=alt.Theta(field="value", type="quantitative"),
        color=alt.Color(field="产品检测结果", type="nominal"),
    )


    #total.drop(index = [total_num],inplace = True)
    st.subheader('数据总表')
    st.dataframe(total)  # Same as st.write(df)  
    st.text('————————————————————————————————————————————————————————————————————————————————————————————————————————')
    st.subheader('各项缺陷统计情况')
    st.dataframe(detail)
    st.text('————————————————————————————————————————————————————————————————————————————————————————————————————————')
    st.text(f'合格率 : {Qualified_Rate}')
    st.altair_chart(bar, use_container_width=True)  
    st.altair_chart(pie_cat, use_container_width=True)
    st.altair_chart(pie_rate, use_container_width=True)

def searchBytime():
    d = st.sidebar.date_input(
    "根据日期查询")
    rows = select("select * from board_detection_results")
    total = pd.DataFrame(
    index=range(1), 
    columns=['主板编号         ','风扇螺丝松动         ','缺少风扇螺丝         ','主板螺丝松动         ','主板螺丝型号不正确         ',
        '缺少主板螺丝         ','风扇接线不正确         ','是否合格         ','检测时间         '])              
    count = 0
    num_Loose_fan_screws = 0
    num_Missing_fan_screws = 0
    num_Loose_board_screws = 0
    num_Board_screw_model_error = 0
    num_Missing_board_screws = 0
    num_Incorrect_fan_wiring = 0
    num_unQualified = 0
    for row in rows:
        if str2date(row[8]) == (str1date(f'{d.year}{d.month}{d.day}')):
            num_Loose_fan_screws +=row[1]
            num_Missing_fan_screws +=row[2]
            num_Loose_board_screws +=row[3]
            num_Board_screw_model_error +=row[4]
            num_Missing_board_screws +=row[5]
            num_Incorrect_fan_wiring +=row[6]
            x = ''.join(re.findall(r'.{5}', str(row[7]))[0])
            if x == "False":
                num_unQualified +=1
            total.loc[count] = [row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8]]
            count+=1

    total_num =  count
    num_Qulified = total_num - num_unQualified
    if (total_num != 0):Qualified_Rate = num_Qulified / total_num
    else: Qualified_Rate = 0
    detail = pd.DataFrame(
        index=range(1), 
        columns=['风扇螺丝松动         ','缺少风扇螺丝           ','主板螺丝松动          ','主板螺丝型号不正确            ',
        '缺少主板螺丝          ','风扇接线不正确          ','不合格       '])           

    detail.loc[0] = [num_Loose_fan_screws,num_Missing_fan_screws,num_Loose_board_screws,num_Board_screw_model_error,
    num_Missing_board_screws,num_Incorrect_fan_wiring,num_unQualified]


    #柱状图
    source = pd.DataFrame({
    '缺陷种类': ['风扇螺丝松动         ','缺少风扇螺丝           ','主板螺丝松动          ','主板螺丝型号不正确            ',
        '缺少主板螺丝          ','风扇接线不正确          '],
    '数量': [num_Loose_fan_screws,num_Missing_fan_screws,num_Loose_board_screws,num_Board_screw_model_error,
    num_Missing_board_screws,num_Incorrect_fan_wiring]
    })

    bar = alt.Chart(source).mark_bar().encode(
        x='数量',
        y='缺陷种类'
    )

    
    #饼状图
    source = pd.DataFrame({"缺陷检测结果": ['风扇螺丝松动         ','缺少风扇螺丝           ','主板螺丝松动          ','主板螺丝型号不正确            ',
        '缺少主板螺丝          ','风扇接线不正确          '], "value": [num_Loose_fan_screws,num_Missing_fan_screws,num_Loose_board_screws,num_Board_screw_model_error,
    num_Missing_board_screws,num_Incorrect_fan_wiring]})

    pie_cat = alt.Chart(source).mark_arc().encode(
        theta=alt.Theta(field="value", type="quantitative"),
        color=alt.Color(field="缺陷检测结果", type="nominal"),
    )



    #合格率饼状图
    source = pd.DataFrame({"产品检测结果": ['合格','不合格'], "value": [num_Qulified,num_unQualified]})

    pie_rate = alt.Chart(source).mark_arc().encode(
        theta=alt.Theta(field="value", type="quantitative"),
        color=alt.Color(field="产品检测结果", type="nominal"),
    )


    #total.drop(index = [total_num],inplace = True)
    st.subheader('数据总表')
    st.dataframe(total)  # Same as st.write(df)  
    st.text('————————————————————————————————————————————————————————————————————————————————————————————————————————')
    st.subheader('各项缺陷统计情况')
    st.dataframe(detail)
    st.text('————————————————————————————————————————————————————————————————————————————————————————————————————————')
    st.text(f'合格率 : {Qualified_Rate}')
    st.altair_chart(bar, use_container_width=True)  
    st.altair_chart(pie_cat, use_container_width=True)
    st.altair_chart(pie_rate, use_container_width=True)

def searchBytime1():
    d = st.sidebar.date_input(
    "根据日期查询")
    rows = select("select * from board_detection_results")
    total = pd.DataFrame(
    index=range(1), 
    columns=['主板编号         ','风扇螺丝松动         ','缺少风扇螺丝         ','主板螺丝松动         ','主板螺丝型号不正确         ',
        '缺少主板螺丝         ','风扇接线不正确         ','是否合格         ','检测时间         '])              
    count = 0
    for row in rows:
        if str2date(row[8]) == (str1date(f'{d.year}{d.month}{d.day}')):
            total.loc[count] = [row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8]]
            count+=1
    st.dataframe(total)    

def searchByperiod():
    a = st.sidebar.date_input(
    "起始时间")
    b = st.sidebar.date_input(
    "结束时间")
    rows = select("select * from board_detection_results")
    total = pd.DataFrame(
    index=range(1), 
    columns=['主板编号         ','风扇螺丝松动         ','缺少风扇螺丝         ','主板螺丝松动         ','主板螺丝型号不正确         ',
        '缺少主板螺丝         ','风扇接线不正确         ','是否合格         ','检测时间         '])              
    count = 0
    num_Loose_fan_screws = 0
    num_Missing_fan_screws = 0
    num_Loose_board_screws = 0
    num_Board_screw_model_error = 0
    num_Missing_board_screws = 0
    num_Incorrect_fan_wiring = 0
    num_unQualified = 0
    for row in rows:
        if (str2date(row[8]) >= (str1date(f'{a.year}{a.month}{a.day}'))) and (str2date(row[8]) <= (str1date(f'{b.year}{b.month}{b.day}'))):
            total.loc[count] = [row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8]]
            num_Loose_fan_screws +=row[1]
            num_Missing_fan_screws +=row[2]
            num_Loose_board_screws +=row[3]
            num_Board_screw_model_error +=row[4]
            num_Missing_board_screws +=row[5]
            num_Incorrect_fan_wiring +=row[6]
            x = ''.join(re.findall(r'.{5}', str(row[7]))[0])
            if x == "False":
                num_unQualified +=1
            count+=1

    total_num =  count
    num_Qulified = total_num - num_unQualified
    if (total_num != 0):Qualified_Rate = num_Qulified / total_num
    else: Qualified_Rate = 0
    detail = pd.DataFrame(
        index=range(1), 
        columns=['风扇螺丝松动         ','缺少风扇螺丝           ','主板螺丝松动          ','主板螺丝型号不正确            ',
        '缺少主板螺丝          ','风扇接线不正确          ','不合格       '])           

    detail.loc[0] = [num_Loose_fan_screws,num_Missing_fan_screws,num_Loose_board_screws,num_Board_screw_model_error,
    num_Missing_board_screws,num_Incorrect_fan_wiring,num_unQualified]


    #柱状图
    source = pd.DataFrame({
    '缺陷种类': ['风扇螺丝松动         ','缺少风扇螺丝           ','主板螺丝松动          ','主板螺丝型号不正确            ',
        '缺少主板螺丝          ','风扇接线不正确          '],
    '数量': [num_Loose_fan_screws,num_Missing_fan_screws,num_Loose_board_screws,num_Board_screw_model_error,
    num_Missing_board_screws,num_Incorrect_fan_wiring]
    })

    bar = alt.Chart(source).mark_bar().encode(
        y='缺陷种类',
        x='数量'
    )

    
    #饼状图
    source = pd.DataFrame({"缺陷检测结果": ['风扇螺丝松动         ','缺少风扇螺丝           ','主板螺丝松动          ','主板螺丝型号不正确            ',
        '缺少主板螺丝          ','风扇接线不正确          '], "value": [num_Loose_fan_screws,num_Missing_fan_screws,num_Loose_board_screws,num_Board_screw_model_error,
    num_Missing_board_screws,num_Incorrect_fan_wiring]})

    pie_cat = alt.Chart(source).mark_arc().encode(
        theta=alt.Theta(field="value", type="quantitative"),
        color=alt.Color(field="缺陷检测结果", type="nominal"),
    )



    #合格率饼状图
    source = pd.DataFrame({"产品检测结果": ['合格','不合格'], "value": [num_Qulified,num_unQualified]})

    pie_rate = alt.Chart(source).mark_arc().encode(
        theta=alt.Theta(field="value", type="quantitative"),
        color=alt.Color(field="产品检测结果", type="nominal"),
    )


    #total.drop(index = [total_num],inplace = True)
    st.subheader('数据总表')
    st.dataframe(total)  # Same as st.write(df)  
    st.text('————————————————————————————————————————————————————————————————————————————————————————————————————————')
    st.subheader('各项缺陷统计情况')
    st.dataframe(detail)
    st.text('————————————————————————————————————————————————————————————————————————————————————————————————————————')
    st.text(f'合格率 : {Qualified_Rate}')
    st.altair_chart(bar, use_container_width=True)  
    st.altair_chart(pie_cat, use_container_width=True)
    st.altair_chart(pie_rate, use_container_width=True)   


def searchByperiod1():
    a = st.sidebar.date_input(
    "起始时间")
    b = st.sidebar.date_input(
    "结束时间")
    rows = select("select * from board_detection_results")
    total = pd.DataFrame(
    index=range(1), 
    columns=['主板编号         ','风扇螺丝松动         ','缺少风扇螺丝         ','主板螺丝松动         ','主板螺丝型号不正确         ',
        '缺少主板螺丝         ','风扇接线不正确         ','是否合格         ','检测时间         '])              
    count = 0
    for row in rows:
        if (str2date(row[8]) >= (str1date(f'{a.year}{a.month}{a.day}'))) and (str2date(row[8]) <= (str1date(f'{b.year}{b.month}{b.day}'))):
            total.loc[count] = [row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8]]
            count+=1
    st.dataframe(total)    

def searchByQuery(query):
    rows = select("select * from board_detection_results"+ query) 
    Loose_fan_screws = select("SELECT Loose_fan_screws from board_detection_results"+ query)
    Missing_fan_screws = select("SELECT Missing_fan_screws from board_detection_results"+ query)
    Loose_board_screws = select("SELECT Loose_board_screws from board_detection_results"+ query)
    Board_screw_model_error = select("SELECT Board_screw_model_error from board_detection_results"+ query)
    Missing_board_screws = select("SELECT Missing_board_screws from board_detection_results"+ query)
    Incorrect_fan_wiring = select("SELECT Incorrect_fan_wiring from board_detection_results"+ query)
    Is_Qualified = select("SELECT Qualified from board_detection_results"+ query)

    total = pd.DataFrame(
            index=range(1), 
            columns=['主板编号         ','风扇螺丝松动         ','缺少风扇螺丝         ','主板螺丝松动         ','主板螺丝型号不正确         ',
            '缺少主板螺丝         ','风扇接线不正确         ','是否合格         ','检测时间         '])  
    count = 0
    for row in rows:
        total.loc[count] = [row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8]]
        count+=1

    num_Loose_fan_screws = count_num(Loose_fan_screws)
    num_Missing_fan_screws = count_num(Missing_fan_screws)
    num_Loose_board_screws = count_num(Loose_board_screws)
    num_Board_screw_model_error = count_num(Board_screw_model_error)
    num_Missing_board_screws = count_num(Missing_board_screws)
    num_Incorrect_fan_wiring = count_num(Incorrect_fan_wiring)
    num_unQualified = count_qualified(Is_Qualified)
    total_num =  count
    num_Qulified = total_num - num_unQualified
    if (total_num != 0):Qualified_Rate = num_Qulified / total_num
    else: Qualified_Rate = 0

    detail = pd.DataFrame(
        index=range(1), 
        columns=['风扇螺丝松动         ','缺少风扇螺丝           ','主板螺丝松动          ','主板螺丝型号不正确            ',
        '缺少主板螺丝          ','风扇接线不正确          ','不合格       '])           

    detail.loc[0] = [num_Loose_fan_screws,num_Missing_fan_screws,num_Loose_board_screws,num_Board_screw_model_error,
    num_Missing_board_screws,num_Incorrect_fan_wiring,num_unQualified]
                
    

    #柱状图
    source = pd.DataFrame({
    '缺陷种类': ['风扇螺丝松动         ','缺少风扇螺丝           ','主板螺丝松动          ','主板螺丝型号不正确            ',
        '缺少主板螺丝          ','风扇接线不正确          '],
    '数量': [num_Loose_fan_screws,num_Missing_fan_screws,num_Loose_board_screws,num_Board_screw_model_error,
    num_Missing_board_screws,num_Incorrect_fan_wiring]
    })

    bar = alt.Chart(source).mark_bar().encode(
        y='缺陷种类',
        x='数量'
    )

    
    #饼状图
    source = pd.DataFrame({"缺陷检测结果": ['风扇螺丝松动         ','缺少风扇螺丝           ','主板螺丝松动          ','主板螺丝型号不正确            ',
        '缺少主板螺丝          ','风扇接线不正确          '], "value": [num_Loose_fan_screws,num_Missing_fan_screws,num_Loose_board_screws,num_Board_screw_model_error,
    num_Missing_board_screws,num_Incorrect_fan_wiring]})

    pie_cat = alt.Chart(source).mark_arc().encode(
        theta=alt.Theta(field="value", type="quantitative"),
        color=alt.Color(field="缺陷检测结果", type="nominal"),
    )



    #合格率饼状图
    source = pd.DataFrame({"产品检测结果": ['合格','不合格'], "value": [num_Qulified,num_unQualified]})

    pie_rate = alt.Chart(source).mark_arc().encode(
        theta=alt.Theta(field="value", type="quantitative"),
        color=alt.Color(field="产品检测结果", type="nominal"),
    )


    #total.drop(index = [total_num],inplace = True)
    st.subheader('数据总表')
    st.dataframe(total)  # Same as st.write(df)  
    st.text('————————————————————————————————————————————————————————————————————————————————————————————————————————')
    st.subheader('各项缺陷统计情况')
    st.dataframe(detail)
    st.text('————————————————————————————————————————————————————————————————————————————————————————————————————————')
    st.text(f'合格率 : {Qualified_Rate}')
    st.altair_chart(bar, use_container_width=True)  
    st.altair_chart(pie_cat, use_container_width=True)
    st.altair_chart(pie_rate, use_container_width=True)

def searchByerro():
    where = " where"
    ad = "and"
    Loose_fan_screws = "Loose_fan_screws>0"
    Missing_fan_screws = "Missing_fan_screws>0"
    Loose_board_screws = "Loose_board_screws>0"
    Board_screw_model_errors = "Board_screw_model_error>0"
    Missing_board_screws = "Missing_board_screws>0"
    Incorrect_fan_wiring = "Incorrect_fan_wiring>0"
    options = st.sidebar.multiselect(
     '缺陷种类',
     ['风扇螺丝松动', '缺少风扇螺丝', '主板螺丝松动', '主板螺丝型号不正确',
     '缺少主板螺丝', '风扇接线不正确'],
     default = '风扇螺丝松动')

    name = {'风扇螺丝松动':Loose_fan_screws,'缺少风扇螺丝':Missing_fan_screws,'主板螺丝松动':Loose_board_screws,
            '主板螺丝型号不正确':Board_screw_model_errors,'缺少主板螺丝':Missing_board_screws,'风扇接线不正确':Incorrect_fan_wiring,}

    count = 0
    query = where
    for i,j in name.items():
        for x in options:
            if x == i:
                if count == 0:
                    query +=" " + " " + j
                    count = 1
                else:
                    query +=" " + ad + " " + j
    searchByQuery(query)



def run_detect():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str,
                        default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.35, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    pic_num = 0
    source = ("图片检测", "视频检测")
    source_index = st.sidebar.selectbox("选择输入", range(
        len(source)), format_func=lambda x: source[x])
    if source_index == 0:
        uploaded_files = st.sidebar.file_uploader(
            "上传图片",accept_multiple_files=True,type=['png', 'jpeg', 'jpg'])
        if uploaded_files is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.image(uploaded_files)
                for uploaded_file in uploaded_files:
                    pic_num += 1
                    picture = Image.open(uploaded_file)
                    picture = picture.save(f'data/images/{uploaded_file.name}')
                #opt.source = f'data/images/{uploaded_file.name}'
                    
        else:
            is_valid = False
    else:
        uploaded_file = st.sidebar.file_uploader("上传视频", type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.video(uploaded_file)
                with open(os.path.join("data", "images", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    #opt.source = f'data/videos/{uploaded_file.name}'
        else:
            is_valid = False

    if is_valid:
        print('valid')
        if st.button('开始检测'): 
            if source_index == 0:
               detect_image(opt,conn)
               with st.spinner(text='Preparing Images'):
                for img in os.listdir(get_detection_folder()):
                    if os.path.splitext(img)[1] == ".jpg":
                       st.image(str(Path(f'{get_detection_folder()}') / img))
                for img in os.listdir(get_detection_folder()):
                    if os.path.splitext(img)[1] == ".png":
                       st.image(str(Path(f'{get_detection_folder()}') / img))
                st.balloons()
            if pic_num == 1:
                for img in os.listdir(get_detection_folder()):
                    with open(str(Path(f'{get_detection_folder()}') / img), "rb") as file:
                        btn = st.download_button(
                                label="Download image",
                                data=file,
                                file_name=str(Path(f'{get_detection_folder()}') / img),
                                #mime="image/png"
                       )
            else:
                detect_video(opt) 
                with st.spinner(text='Preparing Video'):
                    for vid in os.listdir(get_detection_folder()):
                        if os.path.splitext(vid)[1] == ".MP4" or os.path.splitext(vid)[1] == ".mp4":         
                            #change(vid,str(Path(f'{get_detection_folder()}') / vid))                  
                            st.video(str(Path(f'{get_detection_folder()}') / vid))
                            #st.video(path)
                            pass
                    for vid in os.listdir(get_detection_folder()):
                        if os.path.splitext(vid)[1] == ".png":
                           #st.image(str(Path(f'{get_detection_folder()}') / vid))
                           pass
                        #st.balloons()
if __name__ == '__main__':
        
    import shutil
    shutil.rmtree(UPLOAD_FOLDER)
    os.mkdir(UPLOAD_FOLDER)

    st.title('基于图像识别的主板质量检测系统')
    option = st.sidebar.selectbox(
     '选择操作',
     ( '检测','查看数据库'))
    if option == '检测':
        run_detect()
    elif option == '查看数据库':
        optionToDB = st.sidebar.selectbox(
        '选择对数据库的操作',
        ( '查看数据总库',"根据时间查询","根据时间段查询","根据缺陷种类查询"))
        if optionToDB=="查看数据总库":
            search() 
        elif optionToDB=='根据时间段查询':
            searchByperiod()
        elif optionToDB=='根据时间查询':
            searchBytime()
        elif optionToDB=='根据缺陷种类查询':
            searchByerro()
    

    
    
