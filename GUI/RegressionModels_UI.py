import os
import sys
import ast
import time
import threading
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from RegressionModels_form import Ui_Form

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from deepforest import CascadeForestRegressor



class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass



class RegressionModels_UI(QtWidgets.QWidget, Ui_Form):
    
    def __init__(self):
        super(RegressionModels_UI, self).__init__()
        self.setupUi(self)


        self.load_dataset.clicked.connect(self.LoadDataset)
        self.model_select.currentTextChanged.connect(self.UpdateModelParam)
        self.pushButton_2.clicked.connect(self.StartThread)
        self.pushButton_3.clicked.connect(self.SelectSaveFolder)


    def LoadDataset(self):
        path = os.getcwd()  # 获取当前路径
        xlsx_file, ok = QtWidgets.QFileDialog.getOpenFileName(self,"Please select dataset", path, "Excel Files (*.xlsx)" )  # 获取文件路径
        if ok:
            self.xlsx_file_path = xlsx_file
            folder_path, xlsx_file_name = os.path.split(xlsx_file)
            self.xlsx_file_name = xlsx_file_name
            self.folder_path = folder_path
            self.label_4.setText('Excel file has selected: ' + xlsx_file_name)
            self.pushButton_2.setEnabled(True)


    def SelectSaveFolder(self):
        path = os.getcwd()  #获取当前路径
        dir_name = QtWidgets.QFileDialog.getExistingDirectory(self,"Please select save folder", path, QtWidgets.QFileDialog.ShowDirsOnly )  #获取文件夹路径
        if dir_name:
            self.label_12.setText('Save folder has selected: ' + dir_name.split('/')[-1])
            self.save_folder = dir_name

            

    def UpdateModelParam(self):
        if self.model_select.currentText() == 'XGBRegressor':
            self.lineEdit.setText("{'n_estimators': [200, 300, 500]}")
            #self.lineEdit_2.setText("{'n_estimators': 500}")
        elif self.model_select.currentText() == 'RandomForestRegressor':
            self.lineEdit.setText("{'n_estimators': [250, 350, 500], 'max_depth':[10, 30, 50], 'max_features': [4, 6]}")
            #self.lineEdit_2.setText("{'n_estimators': 350, 'max_depth': 30, 'max_features': 6}")
        elif self.model_select.currentText() == 'SVR':
            self.lineEdit.setText("{'C': [0.1, 1, 5, 10]}")
            #self.lineEdit_2.setText("{'C': 1}")
        elif self.model_select.currentText() == 'PLSRegression':
            self.lineEdit.setText("{'n_components': [1, 2, 5, 9]}")
            #self.lineEdit_2.setText("{'n_components': 9}")
        elif self.model_select.currentText() == 'BaggingRegressor':
            self.lineEdit.setText("{'n_estimators': [100, 200, 300, 500]}")
            #self.lineEdit_2.setText("{'n_estimators': 500}")
        elif self.model_select.currentText() == 'LinearRegression':
            self.lineEdit.setText("No model parameters need to screen")
            #self.lineEdit_2.setText("Default")
        elif self.model_select.currentText() == 'AdaBoostRegressor':
            self.lineEdit.setText("{'n_estimators': [50, 100, 200], 'learning_rate':[0.1, 0.5, 1]}")
            #self.lineEdit_2.setText("{'learning_rate': 0.1, 'n_estimators': 100}")
        elif self.model_select.currentText() == 'KNeighborsRegressor':
            self.lineEdit.setText("{'weights':['distance'], 'n_neighbors': [5, 10, 30], 'p':[1, 2, 4]}")
            #self.lineEdit_2.setText("{'n_neighbors': 5, 'p': 1, 'weights': 'distance'}")
        elif self.model_select.currentText() == 'MLPRegressor':
            self.lineEdit.setText("{'solver':['lbfgs'], 'activation':['relu'], 'learning_rate_init':[0.001], 'hidden_layer_sizes':[(50,50), (50,20), (50,50,20)], 'max_iter': [100, 200], 'alpha':[0.001, 0.01]}")
            #self.lineEdit_2.setText("{'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (50, 50, 20), 'learning_rate_init': 0.001, 'max_iter': 200, 'solver': 'lbfgs'}")
        elif self.model_select.currentText() == 'ExtraTreeRegressor':
            self.lineEdit.setText("{'max_depth': [3, 10, 20], 'min_samples_leaf':[5, 10, 20], 'min_samples_split':[10, 30, 50]}")
            #self.lineEdit_2.setText("{'max_depth': 20, 'min_samples_leaf': 5, 'min_samples_split': 10}")
        elif self.model_select.currentText() == 'DecisionTreeRegressor':
            self.lineEdit.setText("{'max_depth': [3, 10, 20], 'min_samples_leaf':[5, 10, 20], 'min_samples_split':[10, 30, 50]}")
            #self.lineEdit_2.setText("{'max_depth': 20, 'min_samples_leaf': 5, 'min_samples_split': 10}")
        elif self.model_select.currentText() == 'CascadeForestRegressor':
            self.lineEdit.setText("{'max_layers': [20, 30], 'n_estimators':[2, 10], 'n_trees':[100, 150]}")
            #self.lineEdit_2.setText("Default")
        else:
            pass


    def StartThread(self):
        self.thread_1 = threading.Thread(target = self.ModelFitAndPredict)
        self.thread_1.setDaemon(True)
        self.thread_1.start()


    def ModelFitAndPredict(self):

        self.pushButton_2.setEnabled(False)
        model_name = self.model_select.currentText()
        print('Model: ' + model_name + '\n')

        train_set_ratio = []
        if self.comboBox.currentIndex() == 0:
            train_set_ratio = [0.7]
        elif self.comboBox.currentIndex() == 1:
            train_set_ratio = [0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]

        print('Loading Dataset...\n')
        df = pd.read_excel(self.xlsx_file_path, sheet_name = int(self.lineEdit_5.text()))

        print('Starting dataset pre-treatment...\n')
        print('Using PCA mode ' + str(self.comboBox_3.currentIndex()+1))
        if self.comboBox_3.currentIndex() == 0:
            df = self.PCA_plan1(df)  
        elif self.comboBox_3.currentIndex() == 1:
            df = self.PCA_plan2(df)
        elif self.comboBox_3.currentIndex() == 2:
            df = self.PCA_plan2(df)

        self.df_info(df)  #显示数据集的信息

        for ratio in train_set_ratio:

            df2 = df

            print('Starting split dataset...\n')
            train_x, train_y, test_x, test_y, all_X, all_Y = self.data_set_split_and_pretreat(df2, test_size=1-ratio)

            print('Training model(It may takes a long time)...\n')
            param_grid = ast.literal_eval(self.lineEdit.text())
            model = self.model_fit(train_x, train_y, param_grid, method = model_name)  #训练好的模型

            y_train_pre = model.predict(train_x)  #预测训练集
            y_test_pre = model.predict(test_x)  #预测测试集
            y_pre = model.predict(all_X)  #预测所有数据

            print('\n>>>>>>Report<<<<<< \n')
            print('模型：' + model_name + '，训练集占比：' + str(ratio) + '，PCA mode ' + str(self.comboBox_3.currentIndex()+1) + '\n')
            print("训练集均方误差(MSE)：", mean_squared_error(train_y, y_train_pre), '\n')
            print("训练集均方根误差(RMSE)：", np.sqrt(mean_squared_error(train_y, y_train_pre)), '\n')
            print("测试集均方误差(MSE)：", mean_squared_error(test_y, y_test_pre), '\n')
            print("测试集均方根误差(RMSE)：", np.sqrt(mean_squared_error(test_y, y_test_pre)), '\n')
            print("决定系数(R^2)：", r2_score(test_y, y_test_pre), '\n')

            # 保存测试集的预测值和真实值至excel文档
            df3 = pd.DataFrame(columns=['数据索引', '测试集的预测值', '测试集的真实值'])
            df3['数据索引'] = test_y.index.array + 2
            df3['测试集的预测值'] = y_test_pre
            df3['测试集的真实值'] = test_y.values
            df3.to_excel(self.save_folder+'\{0}（训练比例{1}, PCA mode {2}, RMSE {3}, R2 {4}）-测试集的预测值和真实值_添加数据索引.xlsx'.format(model_name, ratio, self.comboBox_3.currentIndex()+1, str(np.sqrt(mean_squared_error(test_y, y_test_pre)))[0:8], str(r2_score(test_y, y_test_pre))[0:8]), index=False)
            print('Save file: {0}（训练比例{1}, PCA mode {2}, RMSE {3}, R2 {4}）-测试集的预测值和真实值_添加数据索引.xlsx'.format(model_name, ratio, self.comboBox_3.currentIndex()+1, str(np.sqrt(mean_squared_error(test_y, y_test_pre)))[0:8], str(r2_score(test_y, y_test_pre))[0:8]))

            # 保存所有数据的预测结果至excel
            df4 = pd.DataFrame(columns=['数据索引（对应数据集中的第几条数据）', '所有数据的预测值', '所有数据的真实值'])
            df4['数据索引（对应数据集中的第几条数据）'] = all_Y.index.array + 1
            df4['所有数据的预测值'] = y_pre
            df4['所有数据的真实值'] = all_Y.values
            df4.to_excel(self.save_folder+'\{0}（训练比例{1}, PCA mode {2}）-所有数据的预测值和真实值_添加数据索引.xlsx'.format(model_name, ratio, self.comboBox_3.currentIndex()+1), index=False)
            print('Save file: {0}（训练比例{1}, PCA mode {2}）-所有数据的预测值和真实值_添加数据索引.xlsx'.format(model_name, ratio, self.comboBox_3.currentIndex()+1))

            print('\n\n')

        print('\n\n')

        self.pushButton_2.setEnabled(True)




    def PCA_plan1(self, df, n_components = 4):
        X_40 = df.values[:,int(self.lineEdit_3.text()):(int(self.lineEdit_3.text()) + int(self.lineEdit_4.text()))].astype(float)  # 非稳态数据
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_40 = scaler.fit_transform(X_40)  # 对非稳态数据进行PCA之前，先做归一化处理
        pca = PCA(n_components, random_state = 42)  # 取前4个主成分，设置random_state
        X_p = pca.fit(X_40).transform(X_40)  # 降维后的数据
        print('explained_variance_ratio_: ', pca.explained_variance_ratio_, '\n')  # 解释方差比，看每个主成分携带了多少信息
        # 删除非稳态△吸光度和原先的非稳态数据，改为降维后的数据
        index = np.arange(int(self.lineEdit_3.text()), (int(self.lineEdit_3.text()) + int(self.lineEdit_4.text())))
        df.drop(df.columns[index], axis=1, inplace=True)
        df.insert(loc=int(self.lineEdit_3.text()), column='P1', value = X_p[:,0])

        return df


    def PCA_plan2(self, df, n_components = 9):
        X = df.values[:,0:(int(self.lineEdit_3.text()) + int(self.lineEdit_4.text()))].astype(float)
        #print(X.shape)

        scaler = MinMaxScaler(feature_range=(0, 1))
        X = scaler.fit_transform(X)  # 对特征值做归一化处理
        pca = PCA(n_components, random_state = 42)  # 取前9个主成分，设置random_state
        X_p = pca.fit(X).transform(X)  # 降维后的数据
        print('explained_variance_ratio_: ',pca.explained_variance_ratio_, '\n')  # 解释方差比，看每个主成分携带了多少信息
        # 保留前9个主成分，并返回新的df
        # 删除非稳态△吸光度和原先的非稳态数据，改为降维后的数据
        index = np.arange(0, (int(self.lineEdit_3.text()) + int(self.lineEdit_4.text())))
        df.drop(df.columns[index], axis=1, inplace=True)
        df.insert(loc=0, column='P1', value = X_p[:,0])
        df.insert(loc=1, column='P2', value = X_p[:,1])
        df.insert(loc=2, column='P3', value = X_p[:,2])
        df.insert(loc=3, column='P4', value = X_p[:,3])
        df.insert(loc=4, column='P5', value = X_p[:,4])
        df.insert(loc=5, column='P6', value = X_p[:,5])
        df.insert(loc=6, column='P7', value = X_p[:,6])
        df.insert(loc=7, column='P8', value = X_p[:,7])
        df.insert(loc=8, column='P9', value = X_p[:,8])

        return df


    def PCA_plan3(self, df):

        return df


    # 数据集的统计信息
    def df_info(self, df):
        print(df.head(), '\n')  # 查看前5行数据
        print(df.describe(), '\n')  # 查看数据集的详细信息


    def data_set_split_and_pretreat(self, df, test_size = 0.7):
        train_set,test_set = train_test_split(df, test_size=test_size, random_state=42)  # 训练集和测试集划分

        train_x = train_set.drop('稳态△吸光度', axis = 1)  # 训练集，X
        train_y = train_set['稳态△吸光度'].copy()  # 训练集，y

        test_x = test_set.drop('稳态△吸光度', axis = 1)  # 测试集，X
        test_y = test_set['稳态△吸光度'].copy()  # 测试集，y

        all_x = df.drop('稳态△吸光度', axis = 1) # 所有数据，X
        all_y = df['稳态△吸光度'].copy()  # 所有数据，y

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_x = scaler.fit_transform(train_x)  # 特征值归一化处理
        test_x = scaler.fit_transform(test_x)
        all_x = scaler.fit_transform(all_x)

        return train_x, train_y, test_x, test_y, all_x, all_y


    # 模型选择和训练
    def model_fit(self, train_x, train_y, param_grid, method = 'RandomForestRegressor'):

        if method == 'SVR':
            #model = svm.SVR(C=1.2)
            #model.fit(train_x, train_y)

            # 模型调优
            param_grid = param_grid

            model = svm.SVR()
            # 网格搜索+5折交叉验证
            grid_search  = GridSearchCV(model, param_grid, cv = int(self.lineEdit_6.text()), refit=True,
                                        scoring='neg_mean_squared_error')

            grid_search .fit(train_x, train_y)
            cv_result = grid_search.cv_results_
            for mean_score, params in zip(cv_result ["mean_test_score"], cv_result ["params"]):
                print(np.sqrt(-mean_score), params)
            print('grid_search.best_params_ : ' + str(grid_search.best_params_))
            model = grid_search.best_estimator_  # 最好的模型
            return model


        elif method == 'PLSRegression':
            #model = PLSRegression()
            #model.fit(train_x, train_y)

            # 模型调优
            param_grid = param_grid

            model = PLSRegression()
            # 网格搜索+5折交叉验证
            grid_search  = GridSearchCV(model, param_grid, cv = int(self.lineEdit_6.text()), refit=True,
                                        scoring='neg_mean_squared_error')

            grid_search .fit(train_x, train_y)
            cv_result = grid_search.cv_results_
            for mean_score, params in zip(cv_result ["mean_test_score"], cv_result ["params"]):
                print(np.sqrt(-mean_score), params)
            print('grid_search.best_params_ : ' + str(grid_search.best_params_))
            model = grid_search.best_estimator_  # 最好的模型
            return model

        elif method == 'BaggingRegressor':
            #model = BaggingRegressor()
            #model.fit(train_x, train_y)

            # 模型调优
            param_grid = param_grid

            model = BaggingRegressor()
            # 网格搜索+5折交叉验证
            grid_search  = GridSearchCV(model, param_grid, cv = int(self.lineEdit_6.text()), refit=True,
                                        scoring='neg_mean_squared_error')

            grid_search .fit(train_x, train_y)
            cv_result = grid_search.cv_results_
            for mean_score, params in zip(cv_result ["mean_test_score"], cv_result ["params"]):
                print(np.sqrt(-mean_score), params)
            print('grid_search.best_params_ : ' + str(grid_search.best_params_))
            model = grid_search.best_estimator_  # 最好的模型
            return model

        elif method == 'LinearRegression':
            model = LinearRegression()
            model.fit(train_x, train_y)
            return model
            # 不需要调参

        elif method == 'KNeighborsRegressor':
            #model = KNeighborsRegressor(n_neighbors=10)
            #model.fit(train_x, train_y)

            # 模型调优
            param_grid = param_grid

            model = KNeighborsRegressor()
            # 网格搜索+5折交叉验证
            grid_search  = GridSearchCV(model, param_grid, cv = int(self.lineEdit_6.text()), refit=True,
                                        scoring='neg_mean_squared_error')

            grid_search .fit(train_x, train_y)
            cv_result = grid_search.cv_results_
            for mean_score, params in zip(cv_result ["mean_test_score"], cv_result ["params"]):
                print(np.sqrt(-mean_score), params)
            print('grid_search.best_params_ : ' + str(grid_search.best_params_))
            model = grid_search.best_estimator_  # 最好的模型
            return model

        elif method == 'MLPRegressor':
            #model = MLPRegressor(solver='lbfgs', activation='relu', learning_rate_init=0.001, alpha=0.01, max_iter=200, hidden_layer_sizes=(50,20), random_state=42)
            #model.fit(train_x, train_y)

            # 模型调优
            param_grid = param_grid

            model = MLPRegressor(random_state=42)
            # 网格搜索+5折交叉验证
            grid_search  = GridSearchCV(model, param_grid, cv = int(self.lineEdit_6.text()), refit=True,
                                        scoring='neg_mean_squared_error')

            grid_search .fit(train_x, train_y)
            cv_result = grid_search.cv_results_
            for mean_score, params in zip(cv_result ["mean_test_score"], cv_result ["params"]):
                print(np.sqrt(-mean_score), params)
            print('grid_search.best_params_ : ' + str(grid_search.best_params_))
            model = grid_search.best_estimator_  # 最好的模型
            return model

        elif method == 'XGBRegressor':
            #model = XGBRegressor(n_estimators=500,verbosity=1)
            #model.fit(train_x, train_y)

            # 模型调优
            param_grid = param_grid

            model = XGBRegressor(verbosity=1)
            # 网格搜索+5折交叉验证
            grid_search  = GridSearchCV(model, param_grid, cv = int(self.lineEdit_6.text()), refit=True,
                                        scoring='neg_mean_squared_error')

            grid_search .fit(train_x, train_y)
            cv_result = grid_search.cv_results_
            for mean_score, params in zip(cv_result ["mean_test_score"], cv_result ["params"]):
                print(np.sqrt(-mean_score), params)
            print('grid_search.best_params_ : ' + str(grid_search.best_params_))
            model = grid_search.best_estimator_  # 最好的模型
            return model

        elif method == 'AdaBoostRegressor':
            #model = AdaBoostRegressor(random_state=42)
            #model.fit(train_x, train_y)

            # 模型调优
            param_grid = param_grid

            model = AdaBoostRegressor(random_state=42)
            # 网格搜索+5折交叉验证
            grid_search  = GridSearchCV(model, param_grid, cv = int(self.lineEdit_6.text()), refit=True,
                                        scoring='neg_mean_squared_error')

            grid_search .fit(train_x, train_y)
            cv_result = grid_search.cv_results_
            for mean_score, params in zip(cv_result ["mean_test_score"], cv_result ["params"]):
                print(np.sqrt(-mean_score), params)
            print('grid_search.best_params_ : ' + str(grid_search.best_params_))
            model = grid_search.best_estimator_  # 最好的模型
            return model

        elif method == 'ExtraTreeRegressor':
            #model = ExtraTreeRegressor(random_state=42)
            #model.fit(train_x, train_y)

            # 模型调优
            param_grid = param_grid

            model = ExtraTreeRegressor(random_state=42)
            # 网格搜索+5折交叉验证
            grid_search  = GridSearchCV(model, param_grid, cv = int(self.lineEdit_6.text()), refit=True,
                                        scoring='neg_mean_squared_error')

            grid_search .fit(train_x, train_y)
            cv_result = grid_search.cv_results_
            for mean_score, params in zip(cv_result ["mean_test_score"], cv_result ["params"]):
                print(np.sqrt(-mean_score), params)
            print('grid_search.best_params_ : ' + str(grid_search.best_params_))
            model = grid_search.best_estimator_  # 最好的模型
            return model

        elif method == 'DecisionTreeRegressor':
            #model = DecisionTreeRegressor(random_state=42)
            #model.fit(train_x, train_y)

            # 模型调优
            param_grid = param_grid

            model = DecisionTreeRegressor(random_state=42)
            # 网格搜索+5折交叉验证
            grid_search  = GridSearchCV(model, param_grid, cv = int(self.lineEdit_6.text()), refit=True,
                                        scoring='neg_mean_squared_error')

            grid_search .fit(train_x, train_y)
            cv_result = grid_search.cv_results_
            for mean_score, params in zip(cv_result ["mean_test_score"], cv_result ["params"]):
                print(np.sqrt(-mean_score), params)
            print('grid_search.best_params_ : ' + str(grid_search.best_params_))
            model = grid_search.best_estimator_  # 最好的模型
            return model

        elif method == 'CascadeForestRegressor':
            #model = CascadeForestRegressor(random_state=42)
            #model.fit(train_x, train_y)

            # 模型调优
            param_grid = param_grid

            model = CascadeForestRegressor(random_state=42)
            # 网格搜索+5折交叉验证
            grid_search  = GridSearchCV(model, param_grid, cv = int(self.lineEdit_6.text()), refit=True,
                                        scoring='neg_mean_squared_error')

            grid_search .fit(train_x, train_y)
            cv_result = grid_search.cv_results_
            for mean_score, params in zip(cv_result ["mean_test_score"], cv_result ["params"]):
                print(np.sqrt(-mean_score), params)
            print('grid_search.best_params_ : ' + str(grid_search.best_params_))
            model = grid_search.best_estimator_  # 最好的模型
            return model

        elif method == 'RandomForestRegressor':
            # 模型调优
            param_grid = param_grid

            forest_reg = RandomForestRegressor(random_state=42)
            # 5折交叉验证
            grid_search  = GridSearchCV(forest_reg, param_grid, cv = int(self.lineEdit_6.text()),
                                        scoring='neg_mean_squared_error')

            grid_search .fit(train_x, train_y)
            cv_result = grid_search.cv_results_
            for mean_score, params in zip(cv_result ["mean_test_score"], cv_result ["params"]):
                print(np.sqrt(-mean_score), params)
            print('grid_search.best_params_ : ' + str(grid_search.best_params_))
            model = grid_search.best_estimator_  # 最好的模型
            return model





if __name__ == '__main__': 

    #log_path = os.getcwd()
    #log_file_name = log_path + '/log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
    #sys.stdout = Logger(log_file_name)

    app = QtWidgets.QApplication(sys.argv)
    window = RegressionModels_UI()
    window.show()
    app.exec_()

    




    # 打包方法
    # 先在控制台中输入 chcp 65001

    '''
    包含XGBoost的Pyinstall打包项目出现问题 Cannot find XGBoost Library in the candidate path

    解决步骤：
    1、自己生成一个.py的文件hook-xgboost.py
    里面的内容填写

    from PyInstaller.utils.hooks import collect_all

    datas, binaries, hiddenimports = collect_all("xgboost")

    datas, binaries, hiddenimports = collect_all("deepforest")

    然后将文件复制进目录python路径目录/lib/site-package/_pyinstaller_hooks_contrib/hooks/stdhooks

    2、在使用pyinstall命令时，添加--collect-all "xgboost"（注意引号）

    如pyinstaller --xxxx --collect-all "xgboost"

    '''

    # (pytorch) C:\Users\63459>pyinstaller -F -D E:\我的文档\生活\浙大国际科创中心\自动化有机合成\卢佳敏-数据\LJM_data_treat\LJM_data_treat\Paper_and_new_try\规范代码\RegressionModels_UI.py --collect-all "xgboost"


    #-i 图标路径
    #–icon=图标路径
    #-F 打包成一个exe文件
    #-w 使用窗口，无控制台
    #-c 使用控制台，无窗口
    #-D 创建一个目录，里面包含exe以及其他一些依赖性文件