D = 2 #交易最小单位（天）
C = 1000000. #总起始资金额
alpha = 0.05 #止损点
init_factor = 1e4 # 1e5, 1e6

def write(*x):
    with open('D:\\毕业论文\\无学习率.txt', 'a') as file:
        for item in x:
            file.write(str(item)+' ')
        file.write('\n')
        file.close()


write(C)


from copy import deepcopy 
import os, tensorflow as tf
tf.keras.utils.set_random_seed(42)
os.chdir('D:\\毕业论文')

from 问题情景 import 投资者

person = 投资者()

from loading import tables, columns, files
from numpy import array


tables = list(map(lambda x: x.values.reshape(-1, 241, 5), tables))


Prices = []


for table in tables:
    Price = []
    for idx, day in enumerate(table):
        if idx%D==0:
            bid_price = day[-1, columns.index('收盘')]+0.01
        else:
            close_price = day[-1, columns.index('收盘')]-0.01
            low_price = day.min(axis=0)[columns.index('最低')]-0.01
            open_price = day[0, columns.index('开盘')]-0.01
            LP_init = round((1-alpha)*bid_price, 2)
            Price.append((bid_price, close_price, low_price, open_price, LP_init))
    Prices.append(Price)

def Action(investor, n, bid_price, close_price, low_price, open_price, LP_init):
    n = float(n)
    bid_price = float(bid_price)
    close_price = float(close_price)
    if investor.盈亏比率(n, bid_price, open_price) <= -alpha:
        ask_price = open_price
    else:
        if investor.盈亏比率(n, bid_price, low_price) <= -alpha:
            ask_price = LP_init
            while investor.盈亏比率(n, bid_price, ask_price) <= -alpha or investor.盈亏比率(n, bid_price, ask_price-0.01) > -alpha:
                if investor.盈亏比率(n, bid_price, ask_price) <= -alpha:
                    ask_price += 0.01
                elif investor.盈亏比率(n, bid_price, ask_price-0.01) > -alpha:
                    ask_price -= 0.01
        else:
            ask_price = close_price
    买入总额 = n*100.*bid_price
    卖出总额 = n*100.*ask_price
    return 买入总额+investor.证券买入费用(买入总额), 卖出总额-investor.证券卖出费用(卖出总额)

Prices = list(map(lambda Price: list(map(lambda pair: lambda n: Action(person, n, *pair), Price)), Prices))

# 标号与组合优化
Prices = array(Prices).T


def exploration_policy(Qs):
    return tf.stack(list(map(lambda Q: tf.reshape(tf.random.categorical(logits=tf.stack(Q)[None], num_samples=1), ()), Qs)))

def Wave(X):
	X = X.copy()
	M = X.argmax()
	L = len(X)
	if M<=(L-1)/2:
		slope = 1/(L-1-M)
	elif M>(L-1)/2:
		slope = 1/M
	for idx in range(L):
		X[idx] = -abs(idx-M)*slope
	return X+1

class CO():
    def __init__(self, f):
        self.f = f
        self.Qs = []
        self.N = tf.ones(len(boundaries)).numpy()
        for idx, b in enumerate(boundaries):
            '''
            if idx==init:
                self.Qs.append(init_factor*Wave(array([0.]*(b-1)+[1.])))
            else:
                self.Qs.append(init_factor*Wave(array([1.]+[0.]*(b-1))))
            '''
            self.Qs.append(init_factor*Wave(array([1.]+[0.]*(b-1))))
    def maximize(self, tolerance = 200):
        old_best_value = 0.
        count_tolerance = 0
        old_portfolio = list(map(lambda Q: Q.argmax(), self.Qs))
        bests = [(old_portfolio, old_best_value)]
        while count_tolerance<tolerance:
            print('count tolerance:', count_tolerance)
            A = exploration_policy(self.Qs).numpy()
            r = self.f(A, C)
            print('current try',r)
            bests.append((A, r))
            PF = list(map(lambda Q: Q.argmax(), self.Qs))
            bests.append((PF, self.f(PF ,C)))
            bests = sorted(bests, key=lambda x: x[-1])[-1:]
            self.portfolio, self.best_value = bests[0]
            if r==self.best_value:
                self.N = []
                self.Qs = []
                for idx, b in enumerate(boundaries):
                    self.N.append(tf.ones(b).numpy())
                    ar = array([0.]*b)
                    ar[A[idx]] = 1.
                    self.Qs.append(init_factor*Wave(ar))
            else:
                for idx, a in enumerate(A):
                    self.Qs[idx] *= self.N[idx]
                    ar = tf.zeros(len(self.Qs[idx])).numpy()
                    ar[a] = 1.
                    ar = Wave(ar)
                    ar *= r
                    self.Qs[idx] += ar
                    self.N[idx] += 1.
                    self.Qs[idx] /= self.N[idx]
            if old_best_value==self.best_value:
                count_tolerance += 1
            else:
                count_tolerance = 0
            print('best value so far',self.best_value)
            old_best_value = self.best_value
    def get_portfolio(self):
        return self.portfolio, self.best_value



for idx, day in enumerate(Prices):
    write('Day',idx+1)
    boundaries = [] #记录了长度，从1开始的
    # init = []
    for stock in day:
        volumn = 1
        while stock(volumn)[0]<C:
            volumn += 1
        boundaries.append(volumn)
        # init.append((C-stock(volumn-1)[0]+stock(volumn-1)[-1])/C)
    # init = array(init).argmax()
    n_stocks = len(boundaries)
    def function(portfolio, C):
        C0, feasibility = C, C
        for position in range(n_stocks):
            data = day[position](portfolio[position])
            C -= data[0]
            C += data[-1]
            feasibility -= data[0]
        if feasibility<0:
            return -1.
        else:
            return C/C0-1.
    model = CO(function)
    print('Start Maximizing!')
    model.maximize()
    portfolio = model.get_portfolio()
    write('portfolio:', dict(zip(files, portfolio[0])))
    write('return:', portfolio[-1])
    for position in range(n_stocks):
        data = day[position](portfolio[0][position])
        C -= data[0]
        C += data[-1]
    C = round(C, 2)
    write(C)
