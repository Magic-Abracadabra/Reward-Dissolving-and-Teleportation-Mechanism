D = 2 #交易最小单位（天）
C = 1000000. #总起始资金额
alpha = 0.05 #止损点
beta = 100. #惩罚系数
init_factor = 1.



import os, tensorflow as tf
tf.keras.utils.set_random_seed(42)
os.chdir('D:\\毕业论文')

from 问题情景 import 投资者

person = 投资者()

from loading import tables, columns, files
from numpy import array
#from decimal import Decimal as D

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
        p = []
        for idx, b in enumerate(boundaries):
            '''
            if idx==init:
                p.append([0.]*(b-1)+[init_factor])
            else:
                p.append([init_factor]+[0.]*(b-1))
            '''
            p.append(tf.reshape(tf.keras.activations.softmax(tf.stack(init_factor*Wave(array([1.]+[0.]*(b-1)).astype('float32'))[None])), -1))
        self.p = tf.Variable(initial_value=tf.reshape(tf.ragged.stack(p), -1), dtype=tf.float32, trainable=True)
    def maximize(self, n_steps = 200, lr = 1., n_samples = 1):
        answers = []
        portfolio = tf.cast(tf.reshape(tf.map_fn(tf.argmax, tf.RaggedTensor.from_row_lengths(self.p, boundaries), tf.int64), [-1]), dtype=tf.float32)
        self.best_answer = (portfolio, self.f(portfolio,C))
        answers.append(self.best_answer)
        early_stopping = 0
        while early_stopping < n_steps:
            print('tolerance:',early_stopping)
            l = []
            for _ in range(n_samples):
                X = list(tf.RaggedTensor.from_row_lengths(self.p, boundaries))
                X = tf.cast(tf.stack(list(map(lambda x: tf.reshape(tf.random.categorical(logits=tf.reshape(x, [1,-1]), num_samples=1), []), X))), tf.float32)
                grad = tf.zeros(self.p.shape).numpy()
                for idx in range(len(boundaries)):
                    X_index = int(X[idx].numpy())
                    grad[sum(boundaries[:idx])+X_index] = 1./self.p[sum(boundaries[:idx])+X_index]
                grad = tf.stack(grad)
                f_value = self.f(X,C)
                answers.append((X, f_value))
                grad = grad*f_value
                l.append(grad)
            #print(self.p)
            grad = tf.reduce_mean(l, axis=0)
            grad = list(tf.RaggedTensor.from_row_lengths(grad, boundaries))
            # 投影
            grad = list(map(lambda x: x-tf.reshape(tf.reshape(x, [1,-1])@tf.ones([x.shape[0],1]),[])/(x.shape[0])*tf.ones([x.shape[0]]), grad))
            grad = tf.ragged.stack(grad).values
            trial = self.p+lr*grad
            while tf.reduce_any(trial>1) or tf.reduce_any(trial<0):
                grad /= 2.
                trial = self.p+lr*grad
            self.p = self.p.assign_add(lr*grad)
            portfolio = tf.cast(tf.reshape(tf.map_fn(tf.argmax, tf.RaggedTensor.from_row_lengths(self.p, boundaries), tf.int64), [-1]), dtype=tf.float32)
            answers.append((portfolio, self.f(portfolio,C)))
            answers = sorted(answers, key=lambda x: x[-1])[-1:]
            # 发现新大陆
            if self.best_answer[-1]<answers[0][-1]:
                # 瞬移！
                p_nu = tf.ones(shape=self.p.shape).numpy()
                for idx in range(len(boundaries)):
                    X_index = int(answers[0][0][idx].numpy())
                    p_nu[sum(boundaries[:idx])+X_index] = 0.
                p_nu = list(tf.RaggedTensor.from_row_lengths(p_nu, boundaries))
                p_nu = tf.ragged.stack(list(map(lambda x: tf.keras.activations.softmax(tf.stack(init_factor*Wave(x.numpy())[None])).numpy().flatten(), p_nu))).values
                self.p.assign(p_nu)
                print('Teleportation Succeed!')
                early_stopping = 0
            else:
                early_stopping += 1
            self.best_answer = answers[0]
            print('best return:', self.best_answer[-1].numpy())
    def get_portfolio(self):
        return self.best_answer

DATA = open('投影.txt', 'a')
DATA.write(f'C: {C}\n')
DATA.close()
DATA = open('投影.txt', 'a')
for idx, day in enumerate(Prices):
    DATA.write(f'Day {idx+1}\n')
    DATA.close()
    DATA = open('投影.txt', 'a')
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
        return C/C0-1.+float(feasibility<0)*(feasibility-tf.abs(tf.random.normal([]))-beta)
    model = CO(function)
    print('Start Maximizing!')
    model.maximize()
    portfolio = model.get_portfolio()
    DATA.write(f'portfolio: {dict(zip(files, portfolio[0]))}\n')
    DATA.write(f'return: {portfolio[-1].numpy()}\n')
    DATA.close()
    DATA = open('投影.txt', 'a')
    for position in range(n_stocks):
        data = day[position](portfolio[0][position])
        C -= data[0]
        C += data[-1]
    C = round(C, 2)
    DATA.write(f'C: {C}\n')
    DATA.close()
    DATA = open('投影.txt', 'a')
