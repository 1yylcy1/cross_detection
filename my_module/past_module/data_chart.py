import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#导入声音库

def read_files():
    """该函数用于读取对应文件夹下各txt文件的名字"""
    # path = input("请输入location_center的位置：\n") + '/'
    # project_path=input("请输入你的项目地址：\n")
    # path=project_path+'/data_save/location_center'+'/'
    path = 'data_save/location_center'+'/'
    # path=input('请输入location_center地址：(绝对路径)\n')+'/'
    files = os.listdir(path)


    file_names = []
    for file in files:
        if file.split('.')[-1] == 'txt':  # 如果不是txt文件就跳过
            file_names.append(file)
    return path, file_names


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

def data_art():
    path, files = read_files()
    mixed_file(path, files)
    sum_path = path+"location_center3.txt"  # 保存的整合的txt文件地址
    a = np.loadtxt(sum_path, dtype="str", delimiter=",", unpack=False).reshape(len(open(sum_path, 'rU').readlines()), 3)

    second = a[:, 1]  # 第二列 ——x
    second_list = np.array(second, dtype="float")
    third = a[:, 2]  # 第三列——y
    third = third
    third_list = np.array(third, dtype="float") + 1000

    if len(third)!=0:
        max_num = max(third_list)
        min_num = min(third_list)
        sum = max_num - min_num

        plt.plot(third_list, linestyle='-.', color='r', label='y')
        plt.legend(loc='best')
        plt.plot(second_list, linestyle='--', color='g', label='x')
        plt.legend(loc='best')
        if sum >= 200:
            plt.title("crossing_behaviour", fontproperties="SimHei", fontsize=25)
        else:
            plt.title("normal_behaviour", fontproperties="SimHei", fontsize=25)

        plt.ylabel("coordinates", fontproperties="Kaiti", fontsize=25)
        plt.title("crossing_behaviour", fontproperties="SimHei", fontsize=25)
        plt.savefig(path+'img_art.jpg')
        plt.show()
        if sum>=200:
            # 添加声音
            playsound('data/mp3/crossing_warning_1.mp3')
        #清空上次数据
        import os
        for file_name in os.listdir(path):
            if file_name.endswith('.txt')and file_name!='protect.txt' :
                os.remove(path + file_name)

if __name__=='__main__':
    data_art()