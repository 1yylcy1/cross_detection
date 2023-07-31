import mediapipe as mp
import cv2
import numpy as np
import os
import xgboost as xgb
def model_save_load(model, x_transform):
    # 模型加载
    clf = xgb.XGBClassifier()
    booster = xgb.Booster()
    booster.load_model(model)
    clf._Booster = booster

    # 数据预测
    y_pred = [round(value) for value in clf.predict(x_transform)]
    y_pred_proba = clf.predict_proba(x_transform)
    # print('y_pred：', y_pred)
    # print('y_pred_proba：', y_pred_proba)
    return y_pred,y_pred_proba
def pre(img):
    data=[]
    categories_num = {0: 'crossing',1:'falling',2:'walking'}
    # mp.solutions.drawing_utils用于绘制
    mp_drawing = mp.solutions.drawing_utils

    # 参数：1、颜色，2、线条粗细，3、点的半径
    DrawingSpec_point = mp_drawing.DrawingSpec((0, 255, 0), 2, 2)
    DrawingSpec_line = mp_drawing.DrawingSpec((0, 0, 255), 2, 2)

    # mp.solutions.pose，是人的骨架
    mp_pose = mp.solutions.pose

    # 参数：1、是否检测静态图片，2、姿态模型的复杂度，3、结果看起来平滑（用于video有效），4、检测阈值，5、跟踪阈值
    pose_mode = mp_pose.Pose(static_image_mode=True)


    # image = cv2.imread(img)
    image=img
    # image = cv2.imread(img)
    image_hight, image_width, _ = image.shape
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 处理RGB图像
    results = pose_mode.process(image1)
    # image = cv2.imdecode(np.fromfile(os.path.join(img), dtype=np.uint8), -1)
    '''
    mp_pose.PoseLandmark类中共33个人体骨骼点
    '''
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, DrawingSpec_point, DrawingSpec_line)
        for landmark_id, body_axis in enumerate(results.pose_landmarks.landmark):
            data.append(body_axis.x)
            data.append(body_axis.y)
            data.append(body_axis.z)
        x_pred=np.array([data])
        # print(x_pred)
        y_pred,y_pred_proba=model_save_load('xgboost_classifier_model.model', x_pred)

        outcome=categories_num[y_pred[0]]
        probability = y_pred_proba[0]

        #坐标只能 是整数
        # cv2.putText(image, outcome, (500, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2, 20)
        # cv2.imshow('img',image)
        # cv2.waitKey(0)
        # print('outcome:', outcome)
        #一个是输出结果，一个该结果的概率
        return outcome,probability
    else:
        return 0
if __name__=='__main__':
    image=cv2.imread('walk_1.jpg')
    outcome,probability=pre(image)
    # print("outcome  probability",outcome,probability)
