from my_module import da
import sys
import random
from PyQt5.QtChart import QDateTimeAxis, QValueAxis, QSplineSeries, QChart, QChartView
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import QDateTime, Qt, QTimer


class Demo(QChartView, QChart):
    def __init__(self):
        super(Demo, self).__init__()
        self.setWindowTitle('翻越行为检测轨迹图')
        self.resize(800, 600)
        self.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
        self.chart1()
        self.timer1()

    def chart1(self):
        self.chart = QChart()
        self.series = QSplineSeries()
        # 设置曲线名称
        self.series.setName("head_height")
        # 把曲线添加到QChart的实例中
        self.chart.addSeries(self.series)
        #声明并初始化X轴，Y轴
        self.dtaxisX = QDateTimeAxis()
        self.vlaxisY = QValueAxis()
        # 设置坐标轴显示范围
        self.vlaxisY.setMin(0)
        self.vlaxisY.setMax(2000)
        # 设置X轴时间样式
        self.dtaxisX.setFormat("hh:mm:ss")
        # 设置坐标轴上的格点
        self.dtaxisX.setTickCount(10)
        self.vlaxisY.setTickCount(11)
        # 设置坐标轴名称
        self.dtaxisX.setTitleText("Time")
        self.vlaxisY.setTitleText("Height")
        # 设置网格不显示
        self.vlaxisY.setGridLineVisible(True)
        # 把坐标轴添加到chart中
        self.chart.addAxis(self.dtaxisX, Qt.AlignBottom)
        self.chart.addAxis(self.vlaxisY, Qt.AlignLeft)
        # 把曲线关联到坐标轴
        self.series.attachAxis(self.dtaxisX)
        self.series.attachAxis(self.vlaxisY)
        self.setChart(self.chart)

    def timer1(self):
        # QTimer更新数据
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.Line)
        self.timer.start(500) # 0.5s更新数据

    def Line(self):
        global count
        global x
        global y
        x, y = data_chart.data_sum()
        # 获取当前时间
        bjtime = QDateTime.currentDateTime()
        # 更新X轴坐标
        self.dtaxisX.setMin(QDateTime.currentDateTime().addSecs(-20))
        self.dtaxisX.setMax(QDateTime.currentDateTime().addSecs(0))
        # 产生随即数
        # yint = random.randint(-100, 100)
        yint=y[count]
        count+=1
        # print(yint)
        # print(type(yint))
        # 添加数据到曲线末端
        self.series.append(bjtime.toMSecsSinceEpoch(), yint)


if __name__ == "__main__":
    count =0
    x,y=data_chart.data_sum()
    app = QApplication(sys.argv)
    d = Demo()
    d.show()
    sys.exit(app.exec_())