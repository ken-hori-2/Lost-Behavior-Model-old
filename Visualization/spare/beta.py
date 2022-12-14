import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

## 計算
# a = 46+56+1
# b = 54+44+1
a = 1 # o
b = 1 # x


x = np.linspace(0, 1, 100) #x軸
y = beta.pdf(x, a, b)      #y軸
## 平均値、最頻値
print("mean = ",a/(a+b))
# print("mode = ",(a-1)/(a+b-2))
## グラフ
mean = a/(a+b)
plt.plot(x, y, label = mean)
plt.legend()

# # ベータ分布
# ## ライブラリー読み込み
# from scipy.stats import beta
# import numpy as np
# import matplotlib.pyplot as plt
# ## グラフの表示設定
# plt.style.use('ggplot') #グラフスタイル
# plt.rcParams['figure.figsize'] = [12, 9] # グラフサイズ
# ## 計算
# a = 56+1
# b = 44+1
# x = np.linspace(0, 1, 100) #x軸
# y = beta.pdf(x, a, b)      #y軸
# ## 平均値、最頻値
# print("mean = ",a/(a+b))
# print("mode = ",(a-1)/(a+b-2))
# ## グラフ
# plt.plot(x, y)

plt.show()
