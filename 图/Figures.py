import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

As = ['ε-贪心策略', '无学习率', '投影', '无约束', '奖励溶解']
markers = ['o', 'X', '^', 's', '*']
colors = ['b', 'r', 'g', 'k', 'y']
linestyles = ['-', ':', '--', '-', '-.']
labels_with_enters = ['ε-贪心策略', '无学习率的\nQ-学习算法', '连续松弛技术\n（投影校正）', '连续松弛技术\n（Softmax校正）', '奖励溶解算法']
labels = ['ε-贪心策略', '无学习率的Q-学习算法', '连续松弛技术（投影校正）', '连续松弛技术（Softmax校正）', '奖励溶解算法']

returns = {}
Cs = {}

for A in As:
    with open('D:\\毕业论文\\图\\'+A+'.txt') as info:
        info = info.read().split('\n')    
    returns[A] = list(map(lambda x: float(x.replace('return: ', '')), info[3::4]))[:7]
    Cs[A] = list(map(lambda x: float(x.replace('C: ', ''))/1e6, info[::4]))[:8]

X_returns = range(1, 8)
X_Cs = range(1, 9)


FontProperties = FontProperties(fname='C:\\Windows\\Fonts\\simsun.ttc')

fig, ax = plt.subplots()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

medians = []

for A in As:
    medians.append(sorted(returns[A])[3])

ax.annotate('收益率\n\n', xy=(0, 1), xycoords='axes fraction',
            xytext=(0, -10), textcoords='offset points', fontproperties=FontProperties,
            arrowprops=dict(facecolor='black', arrowstyle='<-'), 
            horizontalalignment='center', verticalalignment='bottom')

ax.bar(labels_with_enters, medians)
plt.xticks(fontproperties=FontProperties)
plt.show()

fig, ax = plt.subplots()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

for idx, A in enumerate(As):
    ax.plot(X_Cs, Cs[A], marker=markers[idx], color=colors[idx], label=labels[idx], linestyle=linestyles[idx])

ax.annotate('', xy=(1, 0), xycoords='axes fraction',
            xytext=(-1, 0), textcoords='offset points', fontproperties=FontProperties,
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            horizontalalignment='center', verticalalignment='top')
ax.annotate('    总资产（$10^6$）\n', xy=(0, 1), xycoords='axes fraction',
            xytext=(0, -10), textcoords='offset points', fontproperties=FontProperties,
            arrowprops=dict(facecolor='black', arrowstyle='<-'), 
            horizontalalignment='center', verticalalignment='bottom')
ax.text(8.4,0.9825,'投资组合\n更换轮数', fontproperties=FontProperties)

plt.legend(prop=FontProperties)
plt.show()


fig, ax = plt.subplots()

for idx, A in enumerate(As):
    ax.plot(X_returns, returns[A], marker=markers[idx], color=colors[idx], label=labels[idx], linestyle=linestyles[idx])

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

ax.annotate('', xy=(1, 0), xycoords='axes fraction',
            xytext=(-1, 0), textcoords='offset points', fontproperties=FontProperties,
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            horizontalalignment='center', verticalalignment='top')
ax.annotate('收益率\n\n', xy=(0, 1), xycoords='axes fraction',
            xytext=(0, -10), textcoords='offset points', fontproperties=FontProperties,
            arrowprops=dict(facecolor='black', arrowstyle='<-'), 
            horizontalalignment='center', verticalalignment='bottom')
ax.text(7.3,-0.0055,'投资组合\n更换轮数', fontproperties=FontProperties)

plt.legend(prop=FontProperties, loc='upper left')
plt.show()
