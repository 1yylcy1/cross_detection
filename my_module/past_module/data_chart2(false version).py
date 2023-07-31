import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#依次读取文件
def read_files():
    """该函数用于读取对应文件夹下各txt文件的名字"""
    # path = input("请输入location_center的位置：\n") + '/'
    # project_path=input("请输入你的项目地址：\n")
    # path=project_path+'/data_save/location_center'+'/'
    path = 'data_save/location_center' + '/'
    # path=input('请输入location_center地址：(绝对路径)\n')+'/'
    files = os.listdir(path)


    file_names = []
    for file in files:
        if file.split('.')[-1] == 'txt':  # 如果不是txt文件就跳过
            file_names.append(file)
    return path, file_names

#合并文件
def mixed_file(path, files):
    """该函数用于合并刚才读取的各文件
    输入：文件路径，read_files()返回的文件名
    输出：一个合并后的文件"""
    content = ''
    for file_name in files:
        with open(path + file_name, 'r', encoding='utf-8') as file:
            content = content + file.read()
            file.close()

    with open(path + 'location_center3.txt', 'a', encoding='utf-8') as file:
        file.write(content)
        content = ''
        file.close()
        return path
#获取下x，y坐标列表（列表元素为float）
def data_sum():
    file=open('data_save/location_center/location_center3.txt')
    data_head_y=[]
    data_head_x=[]
    for line in file.readlines():
        curLine=line.strip().split(",")
        data_head_y.append(int(eval(curLine[2])))
        data_head_x.append(int(eval(curLine[1])))
    return data_head_x,data_head_y
#————————————————————————————————————————————————————————————————————————#
# def deal():
#     third_list,second_list=data_art()
#     if len(third)!=0:
#         max_num = max(third_list)
#         min_num = min(third_list)
#         sum = max_num - min_num
#
#         plt.plot(third_list, linestyle='-.', color='r', label='y')
#         plt.legend(loc='best')
#         plt.plot(second_list, linestyle='--', color='g', label='x')
#         plt.legend(loc='best')
#         if sum >= 80:
#             plt.title("crossing_behaviour", fontproperties="SimHei", fontsize=25)
#         else:
#             plt.title("normal_behaviour", fontproperties="SimHei", fontsize=25)
#
#         plt.ylabel("coordinates", fontproperties="Kaiti", fontsize=25)
#         plt.title("crossing_behaviour", fontproperties="SimHei", fontsize=25)
#         plt.savefig(path+'img_art.jpg')
#         plt.show()
#         #清空上次数据
#         import os
#         for file_name in os.listdir(path):
#             if file_name.endswith('.txt')and file_name!='protect.txt' :
#                 os.remove(path + file_name)

if __name__=='__main__':
    print(data_sum())