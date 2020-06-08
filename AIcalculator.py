from tkinter import *
import tkinter.messagebox
from tkinter import simpledialog
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib
import seaborn as sns
import pandas as pd
from sklearn import svm
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from tkinter.filedialog import askopenfilename

from simple import generateDoc


def iExit():
    iExit = tkinter.messagebox.askyesno("统计机器学习计算器", "Confirm if you want to exit.")  # 询问选择对话窗
    if iExit > 0:
        root.destroy()
        return


def Scientific():
    root.resizable(width=False, height=False)
    root.geometry("1800x530+0+0")


def Standard():
    root.resizable(width=False, height=False)
    root.geometry("400x410")

class AIcal:

    def __init__(self):
        self.operator = ""
        self.dimension = 0
        self.trainx = []
        self.trainy = []
        self.testx = []
        self.testy = []
        self.status = "cal"
        self.mse = dict()

    def btnClick(self,number):
        if self.status == "cal":
            self.operator = self.operator + str(number)
            text_Input.set(self.operator)

    def btnC(self):
        self.operator = ""
        text_Input.set("")
        self.trainx = []
        self.trainy = []
        self.testx = []
        self.testy = []


    def btnCE(self):
        if self.status == "cal":
            if self.operator != "":
                self.operator = self.operator[:-1]
                text_Input.set(self.operator)

    def btnEqual(self):
        if self.status == "cal":
            try:
                text_Input.set(str(eval(self.operator)))
                self.operator = str(eval(self.operator))
            except Exception as e:
                self.operator = ""
                text_Input.set("")
                print(f'Receiving {e}')

    def printx(self):
        string = "输入数据x" + "\n"
        if self.trainx == [] and self.testx == []:
            return string
        if self.trainx != []:
            for i in self.trainx:
                string = string + "\n" + ' '.join(map(str,i))
        if self.testx != []:
            for i in self.testx:
                string = string + "\n" + ' '.join(map(str,i))
        return string

    def printy(self):
        string = "输入数据y" + "\n"
        if self.trainy == [] :
            return string
        else:
            for i in self.trainy:
                string = string + "\n" + str(i)
        return string

    def printpredict(self):
        string = ""
        if self.testy == [] :
            return string
        else:
            for i in self.testy:
                string = string + str(i) + "\n"
        return string

    def btninputrainx(self):
        x = simpledialog.askstring("请输入已知数据x","请输入已知数据x,用逗号分隔")
        if len(list(map(float, list(x.split(","))))) != self.dimension:
            tkinter.messagebox.showerror("维数错误")
        else:
            self.trainx.append( list(map(float, list(x.split(",")))) )
            textbook1.delete("1.0",END)
            textbook1.insert(END, self.printx())


    def btninputrainy(self):
        x = simpledialog.askfloat("请输入已知数据y","请输入已知数据y")
        self.trainy.append(x)
        textbook2.delete("1.0",END)
        textbook2.insert(END, self.printy())

    def btninputestx(self):
        x = simpledialog.askstring("请输入已知数据x", "请输入已知数据x,用逗号分隔")
        if len(list(map(float, list(x.split(","))))) != self.dimension:
            tkinter.messagebox.showerror("维数错误")
        else:
            self.testx.append( list(map(float, list(x.split(",")))) )
            textbook1.delete("1.0",END)
            textbook1.insert(END, self.printx())

    def btndimension(self):
        x = simpledialog.askinteger("请输入数据x的维度","请输入x的维度，为整数")
        self.dimension = x

    def dataset(self):
        self.dimension = 5
        self.trainx = [[3010.00,1888.00,81491.00,4.89,180.92],[3350.00,2195.00,86389.00,16.00,420.39],[3688.00,2531.00,92204.00,19.53,570.25],
                       [3941.00,2799.00,95300.00,21.83,776.71],[4258.00,3054.00,99922.00,23.27,792.43],[4736.00,3358.00,106044.00,22.91,947.70],
                       [5652.00,3905.00,11353.00,26.02,1285.22],[7020.00,4879.00,112110.00,27.72,1783.30],[7859.00,5552.00,108579.00,32.43,2281.95],
                       [9313.00,6386.00,112429.00,38.91,2690.23]]
        self.testx = [[11738.00,8038.00,122645.00,37.38,3169.48],[13176.00,9005.00,113807.00,47.19,2450.14]]
        self.trainy = [231.00,298.00,343.00,401.00,445.00,391.00,554.00,744.00,997.00,1310.00]
        textbook1.delete("1.0", END)
        textbook1.insert(END, self.printx())
        textbook2.delete("1.0",END)
        textbook2.insert(END, self.printy())

    def model(self):
        if self.trainx != [] and self.trainy != []:
            x = np.array(self.trainx)
            y = np.array(self.trainy)
            if self.testx != []:
                test = np.array(self.testx)
                predict = np.append(x,test,axis = 0)
            else:
                predict = x



            #线性回归
            model = LinearRegression().fit(x, y)
            all = model.predict(predict)
            self.testy = ['%.2f' % elem for elem in all.tolist()]
            msl = '%.2f' % mean_squared_error(y, model.predict(x))
            self.mse.setdefault('线性回归', []).append(msl)

            text_lr = "线性回归" + "\n" + "均方误差为：" + str(msl) + "\n" + self.printpredict()
            textbook3.delete("1.0", END)
            textbook3.insert(END, text_lr)

            #Lasso回归
            model = linear_model.Lasso(alpha=0.1).fit(x, y)
            all += model.predict(predict)
            self.testy = ['%.2f' % elem for elem in model.predict(predict).tolist()]
            msl = '%.2f' % mean_squared_error(y, model.predict(x))
            self.mse.setdefault('Lasso回归', []).append(msl)

            text_lasso = "Lasso回归" + "\n" + "均方误差为：" + str(msl) + "\n" + self.printpredict()
            textbook4.delete("1.0", END)
            textbook4.insert(END, text_lasso)


            #SVR
            model = svm.SVR(kernel = 'rbf').fit(x, y)
            all += model.predict(predict)
            self.testy = ['%.2f' % elem for elem in model.predict(predict).tolist()]
            msl = '%.2f' % mean_squared_error(y, model.predict(x))
            self.mse.setdefault('SVR', []).append(msl)

            text_SVR = "SVR" + "\n" + "均方误差为：" + str(msl) + "\n" + self.printpredict()
            textbook5.delete("1.0", END)
            textbook5.insert(END, text_SVR)


            #随机森林
            all += self.xgboost()


            #神经网络
            model = MLPRegressor(random_state=1, max_iter=500).fit(x, y)
            all += model.predict(predict)
            self.testy = ['%.2f' % elem for elem in model.predict(predict).tolist()]
            msl = '%.2f' % mean_squared_error(y, model.predict(x))
            text_nn = "神经网络" + "\n" + "均方误差为：" + str(msl) + "\n" + self.printpredict()
            self.mse.setdefault('神经网络', []).append(msl)

            textbook7.delete("1.0", END)
            textbook7.insert(END, text_nn)

            #all
            all = all/5
            self.testy = ['%.2f' % elem for elem in all.tolist()]
            msl = '%.2f' % mean_squared_error(y, all[0:len(self.trainy)])
            text_all = "平均" + "\n" + "均方误差为：" + str(msl) + "\n" + self.printpredict()
            self.mse.setdefault('平均', []).append(msl)
            textbook8.delete("1.0", END)
            textbook8.insert(END, text_all)


    def xgboost(self):
        if self.trainx != [] and self.trainy != []:
            params = {'booster': 'gbtree',
                  'objective': 'reg:linear',
                  'eval_metric': 'rmse',
                  'gamma': 0.2,
                  'min_child_weight': 1.3,
                  'max_depth': 5,
                  'lambda': 10,
                  'subsample': 0.71,
                  'colsample_bytree': 0.7,
                  'colsample_bylevel': 0.7,
                  'eta': 0.01,
                  'tree_method': 'exact',
                  'seed': 0,
                  'nthread': 12
                  }

            x = np.array(self.trainx)
            y = np.array(self.trainy)
            if self.testx != []:
                test = np.array(self.testx)
                predict = np.append(x, test, axis=0)
            else:
                predict = x

            train_out = xgb.DMatrix(x, label=y)
            test_out = xgb.DMatrix(predict)
            watchlist = [(train_out, 'train')]
            model = xgb.train(params, train_out, num_boost_round=3000, evals=watchlist)


            self.testy = [ '%.2f' % elem for elem in model.predict(test_out) ]
            textbook6.delete("1.0", END)

            msl = '%.2f' % mean_squared_error(y, model.predict(xgb.DMatrix(x)))
            self.mse.setdefault('随机森林', []).append(msl)

            text_rf = "随机森林" +"\n" + "均方误差为：" + str(msl) + "\n" +self.printpredict()

            textbook6.insert(END, text_rf)

            return model.predict(test_out)

    def report(self):
        x1 = np.array(self.trainx)
        y1 = np.array(self.trainy)[:, np.newaxis]
        x = sm.add_constant(self.trainx)
        model = sm.OLS(self.trainy, x)

        combined = np.append(y1,x1,axis=1) # [2159, 10]

        df = pd.DataFrame(combined, columns=['y'] + ['x' + str(i) for i in range(1, x1.shape[1] + 1)])

        print(df.shape)

        corr_matrix = df.corr()
        heat = sns.heatmap(corr_matrix, cmap='PuOr')
        heat.get_figure().savefig("heatmap.png")

        matplotlib.use('Agg')
        matplotlib.style.use('ggplot')
        sns.set()
        sns_plot = sns.pairplot(df, size=2.5)
        sns_plot.savefig("set.png")

        global results
        results = model.fit()

        generateDoc(results, corr_matrix, self.mse)


    def inputdataset(self):
        Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
        filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
        if filename[-4:] == ".txt":
            data = pd.read_csv(filename, sep="\t", header=None)
        elif filename[-4:] == '.csv':
            data = pd.read_csv(filename, header=None)
        data.fillna(data.median(axis=0), inplace=True)


        data1 = data.to_numpy()
        print(data1)

        self.trainx = data1[:,1:].tolist()
        self.trainy = data1[:,0].tolist()

        textbook1.delete("1.0", END)
        textbook1.insert(END, self.printx())

        textbook2.delete("1.0", END)
        textbook2.insert(END, self.printy())

        print(data1.shape)




if __name__ == '__main__':

    root = Tk()
    root.title("统计机器学习计算器")
    root.configure(background="grey")
    root.resizable(width=False, height=False)
    root.geometry("400x405")



    text_Input = StringVar()


    text_Inputx = "输入数据x"
    text_Inputy = "输入数据y"

    text_lr = "线性回归"
    text_lasso = "lasso回归"
    text_SVR = "SVR"
    text_rf = "随机森林"
    text_nn = "神经网络"
    text_all = "平均值"


    cal = AIcal()
    calc = Frame(root)
    calc.grid()

    factor = 2

    txtDisplay = Entry(calc, font=("arial", 25, "bold"), textvariable = text_Input,bg="SlateGray", bd=25, width=10*factor, justify=RIGHT)
    txtDisplay.grid(columnspan=4, pady=1)
    txtDisplay.insert(0, "0")

    text_color = 'WhiteSmoke'

    textbook1 = Text(calc, height = 40, width = 75, bg=text_color)
    textbook1.grid(row = 0, column = 4, columnspan=2, rowspan = 8 )
    textbook1.insert(INSERT,text_Inputx)

    textbook2 = Text(calc, height = 40, width = 12)
    textbook2.grid(row = 0, column = 6, columnspan=1, rowspan = 8 )
    textbook2.insert(INSERT,text_Inputy)

    textbook3 = Text(calc, height = 40, width = 18, bg=text_color)
    textbook3.grid(row = 0, column = 7, columnspan=1, rowspan = 8 )
    textbook3.insert(INSERT,text_lr)

    textbook4 = Text(calc, height = 40, width = 18)
    textbook4.grid(row = 0, column = 8, columnspan=1, rowspan = 8 )
    textbook4.insert(INSERT,text_lasso)

    textbook5 = Text(calc, height = 40, width = 18, bg=text_color)
    textbook5.grid(row = 0, column = 9, columnspan=1, rowspan = 8 )
    textbook5.insert(INSERT,text_SVR)

    textbook6 = Text(calc, height = 40, width = 18)
    textbook6.grid(row = 0, column = 10, columnspan=1, rowspan = 8 )
    textbook6.insert(INSERT,text_rf)

    textbook7 = Text(calc, height = 40, width = 18, bg=text_color)
    textbook7.grid(row = 0, column = 11, columnspan=1, rowspan = 8 )
    textbook7.insert(INSERT,text_nn)

    textbook8 = Text(calc, height=40, width = 18)
    textbook8.grid(row=0, column=12, columnspan=1, rowspan=8)
    textbook8.insert(INSERT, text_all)




    # ===================================== 12345678 ========================================================================

    bottom_font_sz = 12
    font_ = ("arial", 14, "bold")
    fheight = 3
    num_color = 'white'
    # https://blog.csdn.net/chl0000/article/details/7657887

    btn7 = Button(calc, width=bottom_font_sz, height=fheight, font=font_, bg=num_color,
                          bd=4,text="7", command=lambda: cal.btnClick(7)).grid(row=2, column=0, pady=1)
    btn8 = Button(calc, width=bottom_font_sz, height=fheight, font=font_, bg=num_color,
                          bd=4,text="8", command=lambda: cal.btnClick(8)).grid(row=2, column=1, pady=1)
    btn9 = Button(calc, width=bottom_font_sz, height=fheight, font=font_, bg=num_color,
                          bd=4,text="9", command=lambda: cal.btnClick(9)).grid(row=2, column=2, pady=1)


    btn4 = Button(calc, width=bottom_font_sz, height=fheight, font=font_, bg=num_color,
                          bd=4,text="4", command=lambda: cal.btnClick(4)).grid(row=3, column=0, pady=1)
    btn5 = Button(calc, width=bottom_font_sz, height=fheight, font=font_, bg=num_color,
                          bd=4,text="5", command=lambda: cal.btnClick(5)).grid(row=3, column=1, pady=1)
    btn6 = Button(calc, width=bottom_font_sz, height=fheight, font=font_, bg=num_color,
                          bd=4,text="6", command=lambda: cal.btnClick(6)).grid(row=3, column=2, pady=1)

    btn1 = Button(calc, width=bottom_font_sz, height=fheight, font=font_, bg=num_color,
                          bd=4,text="1", command=lambda: cal.btnClick(1)).grid(row=4, column=0, pady=1)
    btn2 = Button(calc, width=bottom_font_sz, height=fheight, font=font_, bg=num_color,
                          bd=4,text="2", command=lambda: cal.btnClick(2)).grid(row=4, column=1, pady=1)
    btn3 = Button(calc, width=bottom_font_sz, height=fheight, font=font_, bg=num_color,
                          bd=4,text="3", command=lambda: cal.btnClick(3)).grid(row=4, column=2, pady=1)


    # ===============================================Standard Buttons========================================================
    btnClear = Button(calc, text=chr(67), width=bottom_font_sz, height=fheight, font=font_,
                      bd=4, bg="grey", command = cal.btnC ).grid(row=1, column=0, pady=1)

    btnback = Button(calc, text="➡", width=bottom_font_sz, height=fheight, font=font_,
                      bd=4, bg="grey", command = cal.btnCE ).grid(row=1, column=1, pady=1)

    btnright = Button(calc, text=")", width=bottom_font_sz, height=fheight, font=font_,
                      bd=4, bg="grey", command=lambda: cal.btnClick(")")).grid(row=1, column=3, pady=1)

    btnleft = Button(calc, text="(", width=bottom_font_sz, height=fheight, font=font_,
                     bd=4, bg="grey", command=lambda: cal.btnClick("(")).grid(row=1, column=2, pady=1)

    btnadd = Button(calc, text="+", width=bottom_font_sz, height=fheight, font=font_,
                    bd=4, bg="grey", command=lambda: cal.btnClick("+")).grid(row=2, column=3, pady=1)

    btnsub = Button(calc, text="-", width=bottom_font_sz, height=fheight, font=font_,
                    bd=4, bg="grey", command=lambda: cal.btnClick("-")).grid(row=3, column=3, pady=1)

    btnmul = Button(calc, text="*", width=bottom_font_sz, height=fheight, font=font_,
                    bd=4, bg="grey", command=lambda: cal.btnClick("*")).grid(row=4, column=3, pady=1)

    btndiv = Button(calc, text="/", width=bottom_font_sz, height=fheight, font=font_,
                    bd=4, bg="grey", command=lambda: cal.btnClick("/")).grid(row=5, column=3, pady=1)

    btnDot = Button(calc, text=".", width=bottom_font_sz, height=fheight, font=font_,
                    bd=4, bg="grey", command=lambda: cal.btnClick(".")).grid(row=5, column=0, pady=1)

    btn0 = Button(calc, text="0", width=bottom_font_sz, height=fheight, font=font_,
                  bd=4, command=lambda: cal.btnClick(0)).grid(row=5, column=1, pady=1)



    btneq = Button(calc, text="=", width=bottom_font_sz, height=fheight, font=font_,
                   bd=4, bg="grey", command= cal.btnEqual).grid(row=5, column=2, pady=1)


    # =============================================== AI Buttons========================================================

    ltext = "输出框"


    btnn = Button(calc, text="x维度", width=bottom_font_sz, height=fheight, font=font_,
                   bd=4, bg="grey",command = cal.btndimension).grid(row=6, column=0, pady=1)

    btntrainx = Button(calc, text="输入已知X", width=bottom_font_sz, height=fheight, font=font_,
                  bd=4, bg="grey",command = cal.btninputrainx).grid(row=6, column=1, pady=1)

    btntrainy = Button(calc, text="输入已知Y", width=bottom_font_sz, height=fheight, font=font_,
                       bd=4, bg="grey",command = cal.btninputrainy).grid(row=6, column=2, pady=1)

    btntestx = Button(calc, text="输入预测X", width=bottom_font_sz, height=fheight, font=font_,
                        bd=4, bg="grey",command = cal.btninputestx).grid(row=6, column=3, pady=1)

    btnlelse = Button(calc, text="测试数据", width=bottom_font_sz, height=fheight, font=font_,
                   bd=4, bg="grey",command = cal.dataset).grid(row=7, column=0, pady=1)

    btninput = Button(calc, text="输入文件", width=bottom_font_sz, height=fheight, font=font_,
                   bd=4, bg="grey",command = cal.inputdataset).grid(row=7, column=1, pady=1)

    btnlinear = Button(calc, text="多模型训练", width=bottom_font_sz, height=fheight, font=font_,
                   bd=4, bg="grey",command = cal.model).grid(row=7, column=2, pady=1)

    btnreport = Button(calc, text="生成报告", width=bottom_font_sz, height=fheight, font=font_,
                        bd=4, bg="grey",command = cal.report).grid(row=7, column=3, pady=1)



    menubar = Menu(calc)

    filemenu = Menu(menubar, tearoff=0)  # 定义一个空菜单单元
    menubar.add_cascade(label="File", menu=filemenu)  # 将上面定义的空菜单命名为`File`，放在菜单栏中，就是装入那个容器中
    filemenu.add_command(label="Standard", command=Standard)  # 在`File`中加入`Standard`小菜单
    filemenu.add_command(label="Machine Learning", command=Scientific)  # 在`File`中加入`Scientific`小菜单
    filemenu.add_separator()  # 这里就是一条分割线
    filemenu.add_command(label="Exit", command=iExit)  # 在`File`中加入`Exit`小菜单

    editmenu = Menu(menubar, tearoff=0)  # 定义一个空菜单单元
    menubar.add_cascade(label="Edit", menu=editmenu)  # 将上面定义的空菜单命名为`Edit`，放在菜单栏中
    editmenu.add_command(label="Cut")  # 在`Edit`中加入`Cut`小菜单
    editmenu.add_command(label="Copy")  # 在`Edit`中加入`Copy`小菜单
    editmenu.add_separator()  # 这里就是一条分割线
    editmenu.add_command(label="Paste")  # 在`Edit`中加入`Paste`小菜单

    helpmenu = Menu(menubar, tearoff=0)  # 定义一个空菜单单元
    menubar.add_cascade(label="Help", menu=helpmenu)  # 将上面定义的空菜单命名为`Help`，放在菜单栏中
    helpmenu.add_command(label="View Help")  # 在`Help`中加入`View Help`小菜单
    helpmenu.add_separator()  # 这里就是一条分割线
    # ================================================================================================================================================================

    root.config(menu=menubar)
    root.mainloop()
