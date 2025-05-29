import matplotlib.pyplot as plt

class 投资者():
	def __init__(self, 交易佣金起底价 = 5., 交易佣金费率 = 2e-5, 过户费率 = 1e-5, 印花税率 = 5e-4):
		self.交易佣金起底价 = 交易佣金起底价 #双向
		self.交易佣金费率 = 交易佣金费率
		self.过户费率 = 过户费率 #双向
		self.印花税率 = 印花税率 #单向
	def 交易佣金(self, 交易额):
		交易额 = float(交易额)
		if 交易额==0.:
			return 0.
		else:
			return round(max(self.交易佣金起底价, 交易额*self.交易佣金费率),2)
	def 过户费(self, 交易额):
		return round(self.过户费率*float(交易额),2)
	def 印花税(self, 卖出额):
		return round(self.印花税率*float(卖出额),2)
	def 证券买入费用(self, 买入总额):
		买入总额 = float(买入总额)
		return self.交易佣金(买入总额)+self.过户费(买入总额)
	def 证券卖出费用(self, 卖出总额):
		卖出总额 = float(卖出总额)
		return self.交易佣金(卖出总额)+self.过户费(卖出总额)+self.印花税(卖出总额)
	def 证券交易费用(self, 买入总额, 卖出总额):
		return self.证券买入费用(float(买入总额))+self.证券卖出费用(float(卖出总额))
	def 盈亏比率(self, 成交量_单位_手, 买价, 卖价):
		成交量_单位_手 = float(成交量_单位_手)
		if 成交量_单位_手==0.:
			return 0.
		else:
			买入总额 = 成交量_单位_手*买价*100.
			卖出总额 = 成交量_单位_手*卖价*100.
			return (卖出总额-self.证券卖出费用(卖出总额))/(买入总额+self.证券买入费用(买入总额))-1.

	def 交易佣金随成交额变化的曲线图(self):
		fig, ax = plt.subplots()
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		节点 = round(s证券交易费用elf.交易佣金起底价/self.交易佣金费率)
		X_max = 节点*2
		Y_max = X_max*self.交易佣金费率
		ax.set_xlim([0, X_max+20000])
		ax.set_ylim([0, Y_max+0.5])
		ax.xaxis.set_ticks_position('bottom')
		ax.yaxis.set_ticks_position('left')
		ax.annotate('', xy=(X_max+20000, 0), xytext=(X_max+19999, 0), arrowprops=dict(facecolor='black', arrowstyle="->"))
		ax.annotate('', xy=(0, Y_max+0.5), xytext=(0, Y_max+0.4), arrowprops=dict(facecolor='black', arrowstyle="->"))
		横坐标 = [0,节点,X_max]
		纵坐标 = [self.交易佣金起底价, self.交易佣金起底价, self.交易佣金费率*X_max]
		虚线横坐标 = [0, 节点]
		ax.plot(横坐标, 纵坐标,'r-')
		ax.plot(虚线横坐标, [0, 节点*self.交易佣金费率],'ko--')
		ax.plot([节点,节点],[0,self.交易佣金起底价], 'b:')
		ax.legend(['交易佣金'])
		plt.title('交易佣金-成交额简化示意图')
		plt.show()

#投资者().交易佣金随成交额变化的曲线图()
