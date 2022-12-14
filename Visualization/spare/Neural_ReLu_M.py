import numpy as np
import math
import matplotlib.pyplot as plt

#パーセプトロン
  #2次元の入力
  #1次元の出力


class neural():

    def __init__(self, Wn):
        self.Wn = Wn
        
    def step(self, x):
        # return 1.0*(x>=0)
        return 1.0*(x>0)

    #sigmoid関数
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    #ReLu関数定義
    def relu(self, x):
        return x*(x>=0)

    def perceptron(self, Xn, B):

        self.Xn = Xn
        
        # ①重み
        # print("重みWn [w1, w2] : ", self.Wn)
        
        # ②バイアス
        self.B = B
        
        # ③入力値の重み付け(ベクトルの内積で計算) + バイアス
        self.XnWn = round(np.dot(self.Xn,self.Wn), 3) + self.B
        
        # ④活性化関数で出力(シグモイド関数)を用いて出力
        "活性化関数"

        self.Output = self.relu(self.XnWn)
        # self.Output = self.step(self.XnWn)
        
        print("Σ (XnWn) : ", self.XnWn)
        return self.Output, self.XnWn

    def visalization(self, XnWn, result):
        #描画
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111) # (211)
        #ReLu関数の入出力データ作成
        # x = np.linspace(-0.5,0.5,5)
        x = np.linspace(-0.5,0.5,500)
        
        "1120 meeting"
        y = self.relu(x)
        # y = self.step(x)
        self.ax.plot(x,y)
        self.ax.scatter(XnWn, result, color="orange")
        # plt.title('ReLu')
        plt.title('Step')
        plt.show()

    def graph(self, index, data_node):

        # カラーマップインスタンス生成
        cm_n = plt.get_cmap("Greens")
        seikika_node = data_node # np.round(preprocessing.minmax_scale(data_node), 3)
        color_maps_node = [cm_n(seikika_node[0]), cm_n(seikika_node[1]), cm_n(seikika_node[2]), cm_n(seikika_node[3])]

        plt.bar(index, data_node, color=color_maps_node, label="ΔS Node")
        plt.title('result bar')
        plt.legend()
        plt.show()

def main():

    # Wn = np.array([1, -0.2]) # -0.2
    
    "なぜこのパラメータにしたのか？2連続発見で0.2は発火しないように設計したからとか"
    # Wn = np.array([1, -0.1])
    Wn = np.array([1, -1])
    print("重みWn [w1, w2] : ", Wn)

    model = neural(Wn)



    "1つのみ出力"
    #入力
    # b, n
    Xn = np.array([3, 3])
    # Xn = np.array([3, 2])
    # Xn = np.array([3, 1])
    print("入力Xn [x1, x2] : ", Xn)
    
    #出力
    B = 0
    T, XnWn = model.perceptron(Xn, B)
    print("出力T : ", abs(T))

    "----- Visualization -----"
    model.visalization(XnWn, T)

    print("\n----------\n")

    #入力
    # T, m, p
    m = 1 # 2 # 1
    p = -1
    Wn = np.array([1, 1]) # , 1])
    print("重みWn [w1, w2, w3] : ", Wn)
    model = neural(Wn)
    
    Xn = np.array([T, m]) # , p])
    print("入力Xn [x1, x2, x3] : ", Xn)
    
    #出力
    result, XnWn = model.perceptron(Xn, p)
    print("出力result : ", abs(result))

    "----- Visualization -----"
    model.visalization(XnWn, result)


    # "複数出力"
    # index = ["O", "A", "B", "C"]
    # data_node = []
    # XnWn_list = []
    
    # "n = 1,2,3,4"
    # # b, n
    # Xn = [np.array([3, 1]), np.array([3, 2]), np.array([3, 3]), np.array([3, 4])]


    # print("入力 : ", Xn)


    # for x in range(len(index)):
        
    #     result, XnWn = model.perceptron(Xn[x])
    #     print(f"出力result {x} : {abs(result)}")
    #     data_node.append(abs(result))
    #     XnWn_list.append(XnWn)

    # print(data_node)

    # model.graph(index, data_node)
    # model.visalization(XnWn_list, data_node)
    

if __name__ == "__main__":
    main()