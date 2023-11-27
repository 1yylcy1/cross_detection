### 友情提示：由于官方给出的yolov5版本会持续更新。为了避免不兼容的问题，建议使用本仓库的yolov5。如果想兼容最新版本的yolov5，自行更改对应的代码即可。
本仓库的yolov5版本为**v5.0**，是直接从官方仓库拉取的，支持训练。

本仓库依赖模型有yolov5s.pt、yolov5m.pt、yolov5l.pt、yolov5x.pt,下载地址：https://github.com/ultralytics/yolov5/releases/tag/v5.0
点击地址后翻到最下面有下载链接，将下载好的模型放在pt文件夹下，运行界面时，**会自动检测已有模型**。

如果模型下载太慢，可以用百度网盘下载，链接：https://pan.baidu.com/s/1sFFWVyidFZZKi76CsKhf6Q?pwd=6666 
提取码：6666

**使用：**
```bash
# conda创建python虚拟环境
conda create -n yolov5_pyqt5 python=3.8
# 激活环境
conda activate yolov5_pyqt5
# 到项目根目录下
cd ./
# 安装依赖文件
pip install -r requirements.txt
# 将下载好的模型放在pt文件夹下
# 运行main.py
python main.py
```
运行*main.py*开启检测界面后**会自动检测已有模型**。ui文件也已上传，可以按照自己的想法更改ui界面。


### 1. 各文件代码说明

result是用来保存实时检测视频结果的
