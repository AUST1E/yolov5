from io import StringIO
import io
from multiprocessing import connection
from pathlib import Path
from tkinter import image_names
from turtle import width
import cv2
from cv2 import VideoCapture
from cv2 import VideoWriter
from matplotlib.pyplot import connect
import streamlit as st
import time
from detect import detect
import os
import sys
import argparse
from PIL import Image
import pyodbc
import pandas as pd
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

    Qualified_Rate = (total_num - num_unQualified)/total_num

    detail = pd.DataFrame(
        index=range(1), 
        columns=['Loose_fan_screws','Missing_fan_screws','Loose_board_screws','Board_screw_model_error',
        'Missing_board_screws','Incorrect_fan_wiring','unQualified'])           

    detail.loc[0] = [num_Loose_fan_screws,num_Missing_fan_screws,num_Loose_board_screws,num_Board_screw_model_error,
    num_Missing_board_screws,num_Incorrect_fan_wiring,num_unQualified]

    # Print results.
    total = pd.DataFrame(
        index=range(3), 
        columns=['Board_num','Loose_fan_screws','Missing_fan_screws','Loose_board_screws','Board_screw_model_error',
        'Missing_board_screws','Incorrect_fan_wiring','Is_Qualified','detect_time'])              
    count = 0
    for row in rows:
        total.loc[count] = [row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8]]
        count+=1
    
    #total.drop(index = [total_num],inplace = True)
    st.subheader('各项缺陷统计情况')
    st.text('————————————————————————————————————————————————————————————————————————————————————————————————————————')
    st.dataframe(detail)
    st.text(f'合格率 : {Qualified_Rate}')
    st.text('————————————————————————————————————————————————————————————————————————————————————————————————————————')
    st.subheader('数据总表')
    st.dataframe(total)  # Same as st.write(df)    

def run_detect():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='weights/yolov5s.pt', help='model.pt path(s)')
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
            detect(opt,conn) 
            if source_index == 0:
               with st.spinner(text='Preparing Images'):
                for img in os.listdir(get_detection_folder()):
                    if os.path.splitext(img)[1] == ".jpg":
                       st.image(str(Path(f'{get_detection_folder()}') / img))
                for img in os.listdir(get_detection_folder()):
                    if os.path.splitext(img)[1] == ".png":
                       st.image(str(Path(f'{get_detection_folder()}') / img))
                st.balloons()
            else:
                #detect(opt) 
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
    if st.sidebar.button("查询数据库"):
       search() 
    run_detect()

    
    