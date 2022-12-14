import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from IPython.display import display

# fp= FontProperties(fname='ipaexg.ttf', size=11)

xRange=(0,1)  # グラフ描画範囲 x
yRange=(0,2)  # グラフ描画範囲 y

plt.figure(figsize=(5,3), dpi=120)

f = lambda t : 6*t - 6*t**2 # ここを書き換え
x = np.linspace(xRange[0], xRange[1], 100)
y = f(x)

plt.plot(x, y, color='orange')

plt.xlim(xRange[0], xRange[1])
plt.xticks(np.linspace(xRange[0], xRange[1], 11))  
# plt.xlabel(r'確率変数 $\theta$（表がでる確率）' , fontproperties=fp )

plt.ylim(yRange[0], yRange[1])
plt.yticks(np.linspace(yRange[0], yRange[1], 5))  
# plt.ylabel(r'確率密度 $f\,(\theta)$', fontproperties=fp )

plt.savefig('test.svg',format = 'svg'); # SVG形式で出力

plt.show()