import cv2
import mediapipe as mp
import os
import del_folder
import numpy as np

#img表示要检测的图片，save_path表示图片保存地址

def body_jpg_detect(img):
    save_path = os.path.abspath('../image-pose.jpg')
    print(f'保存路径为；{save_path}')
    # mp.solutions.drawing_utils用于绘制
    mp_drawing = mp.solutions.drawing_utils

    # 参数：1、颜色，2、线条粗细，3、点的半径
    DrawingSpec_point = mp_drawing.DrawingSpec((0, 255, 0), 2, 2)
    DrawingSpec_line = mp_drawing.DrawingSpec((0, 0, 255), 2, 2)

    # mp.solutions.pose，是人的骨架
    mp_pose = mp.solutions.pose

    # 参数：1、是否检测静态图片，2、姿态模型的复杂度，3、结果看起来平滑（用于video有效），4、检测阈值，5、跟踪阈值
    pose_mode = mp_pose.Pose(static_image_mode=True)

    # file = 'input.jpg'
    file = img
    #image = cv2.imread(file)
    image=cv2.imdecode(np.fromfile(os.path.join(file), dtype=np.uint8),-1)
    image_hight, image_width, _ = image.shape
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 处理RGB图像
    results = pose_mode.process(image1)

    '''
    mp_pose.PoseLandmark类中共33个人体骨骼点
    '''
    if results.pose_landmarks:
        # print(
        #     f'Nose coordinates: ('
        #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_hight})'
        # )
        print(results.pose_landmarks)

    # 绘制
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, DrawingSpec_point, DrawingSpec_line)

    # cv2.imshow('image',image)
    # cv2.waitKey(0)
    #cv2.imwrite(save_path, image)
    cv2.imencode('.jpg', image)[1].tofile(save_path)
    pose_mode.close()

#vedio指视频检测的结果，save_path指视频帧暂时保存地址,out_path表示合帧视频输出地址(带后缀名,如outpy.mp4v)
def body_vedio_detect(vedio,save_path,out_path):
    # mp.solutions.drawing_utils用于绘制
    mp_drawing = mp.solutions.drawing_utils

    # 参数：1、颜色，2、线条粗细，3、点的半径
    DrawingSpec_point = mp_drawing.DrawingSpec((0, 255, 0), 1, 1)
    DrawingSpec_line = mp_drawing.DrawingSpec((0, 0, 255), 1, 1)

    # mp.solutions.pose，是人的骨架
    mp_pose = mp.solutions.pose

    # 参数：1、是否检测静态图片，2、姿态模型的复杂度，3、结果看起来平滑（用于video有效），4、检测阈值，5、跟踪阈值
    pose_mode = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(vedio)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vedio_save_path = save_path
    c = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
        # 对每一帧图片进行处理
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 处理RGB图像
        results = pose_mode.process(image1)

        '''
        mp_holistic.PoseLandmark类中共33个人体骨骼点
        '''

        # 绘制
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, DrawingSpec_point, DrawingSpec_line)

        # 保存每一帧图片
        cv2.imwrite(vedio_save_path + str(c) + '.jpg', image)  # 设置质量
        c += 1
        # 展示每一帧图片
        #cv2.imshow('MediaPipe Pose', image)

        # 窗口按q退出
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
    # 合帧
    l = os.listdir(save_path)  # 帧目录名
    l.sort(key=lambda x: int(x[:-4]))
    print(l)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 创建保存视频的对象，设置编码格式，帧率，图像的宽高等 1080/1920
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), 10, (width, height))

    for i in l:  # i是帧图片的名字(有后缀名)
        img12 = cv2.imread(vedio_save_path + i)
        out.write(img12)

    out.release()
    pose_mode.close()
    cv2.destroyAllWindows()
    cap.release()
    del_folder.del_file(save_path)
body_jpg_detect('D:\文档\大学相关\大学竞赛\高校人工智能大赛\人工智能复赛项目资料\crossing_detection复赛版\smy.jpg')