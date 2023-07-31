# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu
from main_win.win import Ui_mainWindow
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import warnings

warnings.simplefilter("ignore")


import sys
sys.path.append('./my_module')
# print(sys.path)
import json
import torch
import torch.backends.cudnn as cudnn
import os
import time
import cv2

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam
from utils.CustomMessageBox import MessageBox
# LoadWebcam 的最后一个返回值改为 self.cap
from utils.general import check_img_size, check_imshow,non_max_suppression, scale_coords
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device
from utils.capnums import Camera
from dialog.rtsp_win import Window
import matplotlib as mpl
import numpy as np
mpl.use('TkAgg')
from my_module import predict_det

#QThread，它是Qt框架中用于创建多线程的类
class DetThread(QThread):
    #发送np.ndarray类型的信号，通常用于传递图像数据
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    #用于发送dict类型的信号，通常用于传递统计信息
    send_statistic = pyqtSignal(dict)
    # 发送信号：正在检测/暂停/停止/检测结束/错误报告
    #用于发送str类型的信号，通常用于传递消息，统计各类数量
    send_msg = pyqtSignal(str)
    #用于发送int类型的信号，通常用于传递进度百分比
    send_percent = pyqtSignal(int)
    #用于发送str类型的信号
    send_fps = pyqtSignal(str)
    #
    send_result=pyqtSignal(str)



    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = './yolov5s.pt'           # 设置权重
        self.current_weight = './yolov5s.pt'    # 当前权重
        self.source = '0'                       # 视频源
        self.conf_thres = 0.7                  # 置信度
        self.iou_thres = 0.45                   # iou
        self.jump_out = False                   # 跳出循环
        self.is_continue = True                 # 继续/暂停
        self.percent_length = 1000              # 进度条
        self.rate_check = True                  # 是否启用延时
        self.rate = 100                         # 延时HZ
        self.save_fold = './result'             # 保存文件夹
        self.xg_result=None                     #机器学习的检测结果

    #一个装饰器,用于加速训练，节省资源
    @torch.no_grad()
    def run(self,
            imgsz=640,  # inference size (pixels)
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference 用于控制是否对输入进行数据增强
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference(使用 FP16 半精度推理，位数更少可能更快)
            ):

        # Initialize
        try:
            device = select_device(device)#选择设备

            #检查设备是否为cpu，只有当设备类型为 'cpu' 时，half 的值才会被修改为 False
            half &= device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            model = attempt_load(self.weights, map_location=device)  # load FP32 model
            num_params = 0

            """
            model.parameters() 是一个方法调用，它返回模型 model 中的所有可学习参数
            model.parameters() 返回的是一个可迭代对象，可以遍历获取每个参数
            param.numel() 返回的是参数 param 所包含的元素数量
            对于二维张量，元素数量等于行数乘以列数
            每个可学习参数都是一个张量（Tensor）对象，其中包含了一组数值。这些数值表示模型在不同层之间的连接权重、偏置项以及其他学习到的参数
            """
            for param in model.parameters():
                num_params += param.numel()
            """
            stride = int(model.stride.max())
            model.stride 是一个模型（model）的属性，它表示模型中各个层的步幅（stride）。
            而 model.stride.max() 是对步幅的操作，取得所有层中步幅的最大值，并将其转换为整数类型
            """
            stride = int(model.stride.max())  # model stride

            """
            用于检查和调整图像尺寸
            该函数可能会对图像进行缩放、裁剪或填充等操作，以适应模型的输入大小或其他需要
            """
            imgsz = check_img_size(imgsz, s=stride)  # check image size

            """
            如果 model 具有 'module' 属性，那么 names 将被赋值为 model.module.names，
            即取得原始模型对象中的类别名称。否则，直接取得 model.names，即模型对象本身的类别名称
            """
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names

            #是否采用半精度
            if half:
                model.half()  # to FP16

            # Dataloader
            #检查视频来源是否全由数字组成或是否以网络资源形式加载而得
            if self.source.isnumeric() or self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
                #检查设备是否支持图像显示
                view_img = check_imshow()
                cudnn.benchmark = True  # 加速具有恒定图像尺寸的推理过程
                """
                用于从摄像头获取视频流并进行推理
                它接受三个参数：pipe（默认值为'0'）、img_size（默认值为640）和stride（默认值为32）
                pipe参数指定了要使用的视频源，默认为'0'，表示使用默认的摄像头
                它会根据传入的pipe参数来创建一个VideoCapture对象，并设置缓冲区大小为3
                """
                dataset = LoadWebcam(self.source, img_size=imgsz, stride=stride)
                # bs = len(dataset)  # batch_size
            else:
                #加载视频和图片
                dataset = LoadImages(self.source, img_size=imgsz, stride=stride)
                # print(self.source)

            # Run inference
            if device.type != 'cpu':
                """
                创建一个大小为(1, 3, imgsz, imgsz)的全零张量，并将其移动到指定的设备上（使用.to(device)方法）
                调用next(model.parameters())方法获取模型的下一个参数，然后调用.type_as()方法将之前创建的全零张量转换为与参数相同的数据类型
                将模型应用于此全零张量，以便进行一次推理操作 .这个全零张量只是用于初始化模型并确保推理过程正常运行，而不是用于学习或训练模型               
                """
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            count = 0
            # 跳帧检测
            jump_count = 3
            start_time = time.time()
            #iter(dataset) 将视频转换为一个可迭代对象，以便逐帧访问视频的每一帧
            dataset = iter(dataset)
            while True:
                # 手动停止  jump_out默认是停止
                if self.jump_out:
                    self.vid_cap.release()#释放视频捕获资源
                    self.send_percent.emit(0)#发送信号0
                    self.send_msg.emit('停止')#发送信号停止
                    #检查 self 对象是否具有名为 'out' 的属性
                    if hasattr(self, 'out'):
                        self.out.release()#self.out是视频
                    break
                # 临时更换模型，重新加载模型，完成初始化模型的操作，然后将current_weight权重赋值
                if self.current_weight != self.weights:
                    # Load model
                    model = attempt_load(self.weights, map_location=device)  # load FP32 model
                    num_params = 0
                    for param in model.parameters():
                        num_params += param.numel()
                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check image size
                    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
                    if half:
                        model.half()  # to FP16
                    # Run inference
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                    self.current_weight = self.weights
                # 暂停开关
                if self.is_continue:
                    """
                    path 可能表示视频帧的文件路径或标识符。
                    img 可能是代表视频帧的图像数据，可以进行后续的图像处理或分析。
                    im0s 可能是对图像进行了预处理或缩放后的图像数据，用于进一步的处理或显示。
                    self.vid_cap 可能是视频捕获对象或资源，用于后续的视频处理或控制。
                    """
                    path, img, im0s, self.vid_cap = next(dataset)
                    # jump_count += 1
                    # if jump_count % 5 != 0:
                    #     continue
                    count += 1
                    # 每三十帧刷新一次输出帧率,用来计算fps，一秒多少帧
                    if count % 30 == 0 and count >= 30:
                        fps = int(30/(time.time()-start_time))
                        self.send_fps.emit('fps：'+str(fps))#发送字符串，fps信号数据
                        start_time = time.time()
                    if self.vid_cap:
                        """
                        self.percent_length是1000
                        cv2.CAP_PROP_FRAME_COUNT 获取视频捕获对象的总帧数
                        count/self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) 计算的确实是视频处理的百分比。
                        乘以 self.percent_length 的目的是将计算得到的百分比值映射到一个更大的范围上。

                        self.percent_length 是一个缩放因子，将原始的百分比值放大了1000倍
                        这种放大是为了在展示视频处理进度时更精确地呈现，即将进度分成更小的单位。
                        """

                        percent = int(count/self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)*self.percent_length)
                        self.send_percent.emit(percent)#发送int类型数据
                    else:
                        percent = self.percent_length
                    #将names中的每个变量的值都设为0，存储在这个字典中
                    statistic_dic = {name: 0 for name in names}

                    #将输入的图像数据 img 转换为 PyTorch 的张量，并移动到指定的设备（例如 GPU）
                    img = torch.from_numpy(img).to(device)
                    #根据条件 half，将图像张量的数据类型转换为浮点数类型（float）或者半精度浮点数类型（half）
                    img = img.half() if half else img.float()  # uint8 to fp16/32

                    #通过除以 255.0 实现归一化
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)#增加一个维度，新增加的维度是batch size

                    pred = model(img, augment=augment)[0]#获取预测结果中的第一个样本
                    # print(f"yolov5预测结果为{pred}")

                    # Apply NMS  对预测结果进行非极大值抑制
                    """
                    pred 是模型的输出，包含了所有目标的预测结果。
                    self.conf_thres 表示置信度阈值，低于该阈值的预测会被过滤掉。
                    self.iou_thres 是重叠度阈值，用于判断两个边界框是否相交的程度。
                    classes 是一个列表，表示希望保留的类别列表，其他类别的预测会被过滤掉。
                    agnostic_nms 是一个布尔值，用于指定是否使用类别不可知的 NMS。
                    当为 True 时，NMS 不考虑物体的类别，只关注边界框之间的重叠度；当为 False 时，NMS 将按照物体的类别进行分组，在每个类别内部应用 NMS。
                    max_det 是一个整数，表示在应用 NMS 之后，每个图像最多保留的预测边界框数目
                    """
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms, max_det=max_det)
                    # Process detections
                    #pred 是经过预测后的结果。循环 for i, det in enumerate(pred) 遍历了每个图像的预测结果
                    ##det是一个tensor，im0是一个二维数组array
                    for i, det in enumerate(pred):  # detections per image
                        im0 = im0s.copy()#img0  原size图片
                        #frame 变量表示当前处理的是视频源中的第几帧
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)#p是每一帧视频路径
                        #如果有预测结果
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                c = int(cls)  # integer class
                                statistic_dic[names[c]] += 1
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                # im0 = plot_one_box_PIL(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)  # 中文标签画框，但是耗时会增加

                                # 中心点坐标(x0, y0)
                                x0 = (int(xyxy[0].item()) + int(xyxy[2].item())) / 2
                                y0 = (int(xyxy[1].item()) + int(xyxy[3].item())) / 2
                                # print("坐标是",(x0,y0))
                                class_index = cls  # 获取属性
                                object_name = names[int(cls)]  # 获取标签名
                                self.xg_result=object_name
                                # if object_name == 'person':
                                # flocation.write(object_name + ',' + str(x0) + ',' + str(y0) + '\n')
                                #给物体画框框
                                # if 'person' in label:
                                plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                                    line_thickness=line_thickness)

                    #predict_det.pre是机器学习模型
                    predict_det.pre(im0)#a是结果
                    # print('xg_result:',self.xg_result)
                    if self.xg_result:
                        self.send_result.emit(self.xg_result)
                    else:
                        continue
                    # if a !=0:
                    #     cv2.putText(im0, a, (int(x0), int(y0)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255),2, 20)





                    # 控制视频发送频率,self.rate_check：是否启用延时
                    if self.rate_check:
                        time.sleep(1/self.rate)
                    self.send_img.emit(im0)#发送im0信号，二维数组
                    self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])

                    #发送字典信息
                    self.send_statistic.emit(statistic_dic)
                    # 如果自动录制
                    if self.save_fold:
                        """
                        os.makedirs用于递归地创建目录，如果目录已经存在，则不会引发错误,路径不存在时自动创建该目录
                        exist_ok=True是os.makedirs函数的可选参数，表示如果目录已经存在，不会引发错误。
                        如果设置为False（默认值），则如果目录已经存在，os.makedirs将引发FileExistsError异常
                        """
                        os.makedirs(self.save_fold, exist_ok=True)  # 路径不存在，自动保存
                        # 如果输入是图片(图片的下一帧不存在，故为none)
                        if self.vid_cap is None:
                            save_path = os.path.join(self.save_fold,
                                                     time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                   time.localtime()) + '.jpg')
                            #保存图片
                            cv2.imwrite(save_path, im0)
                        else:
                            if count == 1:  # 第一帧时初始化录制
                                # 以视频原始帧率进行录制
                                ori_fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
                                if ori_fps == 0:
                                    ori_fps = 25
                                """
                                分辨率：
                                width = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                """

                                width, height = im0.shape[1], im0.shape[0]
                                save_path = os.path.join(self.save_fold, time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + '.mp4')
                                #cv2.VideoWriter创建一个视频写入对象self.out，设置保存路径、视频的编码格式为MP4，只有在第一帧的时候才创建该对象
                                self.out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), ori_fps,
                                                           (width, height))
                            self.out.write(im0)
                    #如果进度条满了
                    if percent == self.percent_length:
                        print(count)#输出当前帧数
                        self.send_percent.emit(0)#发送进度条信号
                        self.send_msg.emit('检测结束')#发送字符串信号
                        #如果存在视频资源对象，则进行视频资源的释放
                        if hasattr(self, 'out'):
                            self.out.release()
                        # 正常跳出循环(它是对视频每一帧进行检测)
                        break
        except Exception as e:
            self.send_msg.emit('%s' % e)



class MainWindow(QMainWindow, Ui_mainWindow):
    # result_show=None
    def clickButton(self):
        print()

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False
        # win10的CustomizeWindowHint模式，边框上面有一段空白。
        # 不想看到顶部空白可以用FramelessWindowHint模式，但是需要重写鼠标事件才能通过鼠标拉伸窗口，比较麻烦
        # 不嫌麻烦可以试试, 写了一半不想写了，累死人
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint )
        # self.setWindowFlags(Qt.FramelessWindowHint)
        # 自定义标题栏按钮
        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)
        self.closeButton.clicked.connect(self.close)

        #页面切换
        self.pushButton_2.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
        self.pushButton_3.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))

        # 定时清空自定义状态栏上的文字
        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        # 自动搜索模型
        self.comboBox.clear()
        self.pt_list = os.listdir('./pt')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./pt/'+x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        # yolov5线程
        self.det_thread = DetThread()                                   #创建一个yolov5线程对象
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type           # 权重
        self.det_thread.source = '0'                                    # 默认打开本机摄像头，无需保存到配置文件
        self.det_thread.percent_length = self.progressBar.maximum()     #进度条长度
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.raw_video))
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.out_video))
        self.det_thread.send_statistic.connect(self.show_statistic)    #统计各类检测数量
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))   #接受字符串信号
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        # self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))
        self.det_thread.send_result.connect(lambda x:self.result_sh_label.setText(x))#显示输出结果

        self.fileButton.clicked.connect(self.open_file)
        self.cameraButton.clicked.connect(self.chose_cam)
        self.rtspButton.clicked.connect(self.chose_rtsp)

        self.runButton.clicked.connect(self.run_or_continue)
        self.stopButton.clicked.connect(self.stop)

        self.comboBox.currentTextChanged.connect(self.change_model)
        # self.comboBox.currentTextChanged.connect(lambda x: self.statistic_msg('模型切换为%s' % x))
        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))
        self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))

        self.checkBox.clicked.connect(self.checkrate)
        self.saveCheckBox.clicked.connect(self.is_save)
        self.load_setting()
        # self.pushButton.clicked.connect(self.clickButton)





    # def result_show(self):
    #     # 显示结果
    #     self.result_sh_label.setText(self.det_thread.xg_result)
    #     self.fps_label.setAlignment(Qt.AlignCenter)
    #     self.setStyleSheet("QLabel{color:rgb(225,0,0,255);font-size:15px;font-weight:normal;font-family:Arial;}")



    #寻找模型文件
    def search_pt(self):
        pt_list = os.listdir('./pt')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    def is_save(self):
        if self.saveCheckBox.isChecked():
            # 选中时
            self.det_thread.save_fold = './result'
        else:
            self.det_thread.save_fold = None

    def checkrate(self):
        if self.checkBox.isChecked():
            # 选中时
            self.det_thread.rate_check = True
        else:
            self.det_thread.rate_check = False

    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.67:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.closeButton, title='提示', text='请稍等，正在加载rtsp视频流', time=1000, auto=True).exec_()
            self.det_thread.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.statistic_msg('加载rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.statistic_msg('%s' % e)

    def chose_cam(self):
        try:
            self.stop()
            # MessageBox的作用：留出2秒，让上一次摄像头安全release
            MessageBox(
                self.closeButton, title='提示', text='请稍等，正在检测摄像头设备', time=2000, auto=True).exec_()
            # 自动检测本机有哪些摄像头
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.cameraButton.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            ''')

            for cam in cams:
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            x = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).x()
            y = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).y()
            y = y + self.cameraButton.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec_(pos)
            if action:
                self.det_thread.source = action.text()
                self.statistic_msg('加载摄像头：{}'.format(action.text()))
        except Exception as e:
            self.statistic_msg('%s' % e)

    # 导入配置文件
    def load_setting(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            check = 0
            savecheck = 0
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "check": check,
                          "savecheck": savecheck
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                check = 0
                savecheck = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                check = config['check']
                savecheck = config['savecheck']
        self.confSpinBox.setValue(iou)
        self.iouSpinBox.setValue(conf)
        self.rateSpinBox.setValue(rate)
        self.checkBox.setCheckState(check)
        self.det_thread.rate_check = check          # 是否启用延时
        self.saveCheckBox.setCheckState(savecheck)
        self.is_save()                              # 是否自动保存

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x*100))
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x/100)
            self.det_thread.conf_thres = x/100
        elif flag == 'iouSpinBox':
            self.iouSlider.setValue(int(x*100))
        elif flag == 'iouSlider':
            self.iouSpinBox.setValue(x/100)
            self.det_thread.iou_thres = x/100
        elif flag == 'rateSpinBox':
            self.rateSlider.setValue(x)
        elif flag == 'rateSlider':
            self.rateSpinBox.setValue(x)
            self.det_thread.rate = x * 10
        else:
            pass

    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)
        # self.qtimer.start(3000)   # 3秒后自动清除

    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)
        if msg == "检测结束":
            self.saveCheckBox.setEnabled(True)

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.statistic_msg('模型切换为%s' % x)

    def open_file(self):
        # source = QFileDialog.getOpenFileName(self, '选取视频或图片', os.getcwd(), "Pic File(*.mp4 *.mkv *.avi *.flv "
        #                                                                    "*.jpg *.png)")
        config_file = 'config/fold.json'
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, '选取视频或图片', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                          "*.jpg *.png)")
        if name:
            self.det_thread.source = name
            self.statistic_msg('加载文件：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            # 切换文件后，上一次检测停止
            self.stop()

    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()

    # 继续/暂停
    def run_or_continue(self):
        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.saveCheckBox.setEnabled(False)
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
            source = os.path.basename(self.det_thread.source)
            source = '摄像头设备' if source.isnumeric() else source
            self.statistic_msg('正在检测 >> 模型：{}，文件：{}'.
                               format(os.path.basename(self.det_thread.weights),
                                      source))
        else:
            self.det_thread.is_continue = False
            self.statistic_msg('暂停')

    # 退出检测循环
    def stop(self):
        self.det_thread.jump_out = True
        self.saveCheckBox.setEnabled(True)

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
            # QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        # self.setCursor(QCursor(Qt.ArrowCursor))

    #静态方法可以通过类名直接调用，无需创建类的实例。它们与类相关联，而不是与实例相关联
    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # 保持纵横比
            # 找出长边
            if iw > ih:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    # 实时统计
    def show_statistic(self, statistic_dic):
        try:
            self.resultWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' '+str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            self.resultWidget.addItems(results)

        except Exception as e:
            print(repr(e))

    def closeEvent(self, event):
        # 如果摄像头开着，先把摄像头关了再退出，否则极大可能可能导致检测线程未退出
        self.det_thread.jump_out = True
        # 退出时，保存设置
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.confSpinBox.value()
        config['conf'] = self.iouSpinBox.value()
        config['rate'] = self.rateSpinBox.value()
        config['check'] = self.checkBox.checkState()
        config['savecheck'] = self.saveCheckBox.checkState()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        MessageBox(
            self.closeButton, title='提示', text='请稍等，正在关闭程序', time=2000, auto=True).exec_()
        sys.exit(0)
def main():
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())


if __name__=='__main__':
    main()