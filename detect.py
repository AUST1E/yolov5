import argparse
import time
from pathlib import Path
from itertools import chain
import cv2
import torch
import torch.backends.cudnn as cudnn
import pymssql
import datetime
import matplotlib
import os
from matplotlib import pyplot as plt
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #允许副本存在，忽略报错

def detect_image(opt,connect,save_img=False):
    global cls
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    #connect = pymssql.connect('122.112.167.98', 'sa', 'Augustine12!', 'WHSJ_Computer_Board')  # 建立连接
    if connect:
        print("Database connection succeeded!")

    cursor = connect.cursor()  # 创建一个游标对象,python里的sql语句都要通过cursor来执行
    now = datetime.datetime.now()
    otherStyleTime = now.strftime("%Y_%m_%d_%H_%M")

    fileList = os.listdir(source)  # 待修改文件夹
    Board_nums = {}
    i = 0
    for fileName in fileList:  # 遍历文件夹中所有文件
        Board_nums[i] = fileName.split(".")[0]
        i += 1


    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    t = 0
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            cursor.execute("insert into board_detection_results values('{0}','0','0','0','0','0','0','1','{1}')".format(
                str(Board_nums[t]), otherStyleTime))  # 执行sql语句
            connect.commit()  # 提交

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results

                for c in det[:, -1].unique():

                    if names[int(c)] == '1 Loose fan screws' or names[int(c)] == '2 fan screw model error' or \
                            names[int(c)] == '3 Missing fan screws' or names[
                        int(c)] == '5 Loose board screws' or names[int(c)] == '6 board screw model error' or \
                            names[int(c)] == '7 Missing board screws' or names[
                        int(c)] == '9 Incorrect fan wiring' or names[
                        int(c)] == '11 Loose fan fixing screws' or names[
                        int(c)] == '12 Fan fixing screw model error' or names[
                        int(c)] == '13 Missing fan fixing screws':
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    else:
                        continue

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    if names[int(c)] == '1 Loose fan screws':
                        cursor.execute(
                            "update board_detection_results set Loose_fan_screws = Loose_fan_screws+{1} WHERE Board_num = '{0}' update "
                            "board_detection_results set Qualified = 0 WHERE Board_num = '{0}'".format(
                                Board_nums[t], n))  # 执行sql语句
                        connect.commit()  # 提交
                    elif names[int(c)] == '3 Missing fan screws':
                        cursor.execute(
                            "update board_detection_results set Missing_fan_screws = Missing_fan_screws+{1} WHERE Board_num = '{0}'update "
                            "board_detection_results set Qualified = 0 WHERE Board_num = '{0}'".format(
                                Board_nums[t], n))
                        connect.commit()  # 提交
                    elif names[int(c)] == '5 Loose board screws':
                        cursor.execute(
                            "update board_detection_results set Loose_board_screws = Loose_board_screws+{1} WHERE Board_num = '{0}'update "
                            "board_detection_results set Qualified = 0 WHERE Board_num = '{0}'".format(
                                Board_nums[t], n))
                        connect.commit()  # 提交
                    elif names[int(c)] == '6 board screw model error':
                        cursor.execute(
                            "update board_detection_results set Board_screw_model_error = Board_screw_model_error+{1} WHERE Board_num = '{0}' "
                            "update board_detection_results set Qualified = 0 WHERE Board_num = '{0}'".format(Board_nums[t], n))
                        connect.commit()  # 提交
                    elif names[int(c)] == '7 Missing board screws':
                        cursor.execute(
                            "update board_detection_results set Missing_board_screws = Missing_board_screws+{1} WHERE Board_num = '{0}'update "
                            "board_detection_results set Qualified = 0 WHERE Board_num = '{0}'".format(
                                Board_nums[t], n))
                        connect.commit()  # 提交
                    elif names[int(c)] == '9 Incorrect fan wiring':
                        cursor.execute(
                            "update board_detection_results set Incorrect_fan_wiring = Incorrect_fan_wiring+{1} WHERE Board_num = '{0}'update "
                            "board_detection_results set Qualified = 0 WHERE Board_num = '{0}'".format(
                                Board_nums[t], n))
                        connect.commit()  # 提交
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_img or view_img:  # Add bbox to image
                        if names[int(cls)] == '1 Loose fan screws' or names[int(cls)] == '2 fan screw model error' or \
                                names[int(cls)] == '3 Missing fan screws' or names[
                            int(cls)] == '5 Loose board screws' or names[int(cls)] == '6 board screw model error' or \
                                names[int(cls)] == '7 Missing board screws' or names[
                            int(cls)] == '9 Incorrect fan wiring' or names[
                            int(cls)] == '11 Loose fan fixing screws' or names[
                            int(cls)] == '12 Fan fixing screw model error' or names[
                            int(cls)] == '13 Missing fan fixing screws':
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        else:
                            continue
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            t += 1
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)


    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    # sql = 'select Defects_Dnum from Defects_nums{0}'.format(otherStyleTime)
    # cursor.execute(sql)  # 执行sql指令
    # fig = plt.figure(figsize=(15, 8))
    # y1 = cursor.fetchall()  # 获取sql数据
    # y = list(chain.from_iterable(y1))
    # print(y)
    # x = ('风扇螺丝松动', '缺少风扇螺丝', '主板螺丝松动', '主板螺丝型号不正确', '缺少主板螺丝',  '风扇接线不正确')
    # plt.bar(x, y, 0.4, color=['r', 'g', 'b', 'c', 'm', 'y'])
    # plt.ylim(0, 20)
    # plt.title("主板质量检测报表")
    # plt.xlabel("缺陷名称")
    # plt.ylabel("缺陷数量")
    # plt.legend()
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.savefig("test_1.png")
    cursor.close()  # 关闭游标
    #connect.close()  # 关闭连接


def detect_video(opt,save_img=False):
    global cls
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    #connect = pymssql.connect('122.112.167.98', 'sa', 'Augustine12!', 'WHSJ_Computer_Board')  # 建立连接






    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    t = 0
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh



            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results

                for c in det[:, -1].unique():

                    if names[int(c)] == '1 Loose fan screws' or names[int(c)] == '2 fan screw model error' or \
                            names[int(c)] == '3 Missing fan screws' or names[
                        int(c)] == '5 Loose board screws' or names[int(c)] == '6 board screw model error' or \
                            names[int(c)] == '7 Missing board screws' or names[
                        int(c)] == '9 Incorrect fan wiring' or names[
                        int(c)] == '11 Loose fan fixing screws' or names[
                        int(c)] == '12 Fan fixing screw model error' or names[
                        int(c)] == '13 Missing fan fixing screws':
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    else:
                        continue

               
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_img or view_img:  # Add bbox to image
                        if names[int(cls)] == '1 Loose fan screws' or names[int(cls)] == '2 fan screw model error' or \
                                names[int(cls)] == '3 Missing fan screws' or names[
                            int(cls)] == '5 Loose board screws' or names[int(cls)] == '6 board screw model error' or \
                                names[int(cls)] == '7 Missing board screws' or names[
                            int(cls)] == '9 Incorrect fan wiring' or names[
                            int(cls)] == '11 Loose fan fixing screws' or names[
                            int(cls)] == '12 Fan fixing screw model error' or names[
                            int(cls)] == '13 Missing fan fixing screws':
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        else:
                            continue
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            t += 1
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)


    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    # sql = 'select Defects_Dnum from Defects_nums{0}'.format(otherStyleTime)
    # cursor.execute(sql)  # 执行sql指令
    # fig = plt.figure(figsize=(15, 8))
    # y1 = cursor.fetchall()  # 获取sql数据
    # y = list(chain.from_iterable(y1))
    # print(y)
    # x = ('风扇螺丝松动', '缺少风扇螺丝', '主板螺丝松动', '主板螺丝型号不正确', '缺少主板螺丝',  '风扇接线不正确')
    # plt.bar(x, y, 0.4, color=['r', 'g', 'b', 'c', 'm', 'y'])
    # plt.ylim(0, 20)
    # plt.title("主板质量检测报表")
    # plt.xlabel("缺陷名称")
    # plt.ylabel("缺陷数量")
    # plt.legend()
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.savefig("test_1.png")
    #connect.close()  # 关闭连接


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s.pt',
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.50, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.10, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect_image()
                strip_optimizer(opt.weights)
        else:
            detect_image()
