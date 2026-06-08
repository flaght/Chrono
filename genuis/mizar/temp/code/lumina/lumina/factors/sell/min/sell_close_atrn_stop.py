# -*- encoding:utf-8 -*-
"""
    卖出择时示例因子： 较小利润值 < 买入后最大收益价格 - 今日价格 < 较大利润值 －> 止盈卖出
    只做为单边止盈因子使用，作为利润保护因子使用
"""
from lumina.factors.sell.fixes import FactorSellIB, ESupportDirection
import pdb

class FactorCloseAtrNStop(FactorSellIB):
    """示例利润保护因子(止盈)因子"""

    def _init_self(self, **kwargs):
        """kwargs中可选参数close_atr_n: 保护利润止赢倍数"""

        self.close_atr_n = kwargs.get('close_atr_n', 3)
        if 'close_atr_n' in kwargs:
            # 设置保护利润止赢倍数
            self.close_atr_n = kwargs['close_atr_n']
            self.sell_type_extra = '{}:close_atr_n={}'.format(
                self.__class__.__name__, self.close_atr_n)

    def support_direction(self):
        """单日最大跌幅n倍atr(止损)因子支持两个方向"""
        return [
            ESupportDirection.DIRECTION_CAll.value,
            ESupportDirection.DIRECTION_PUT.value
        ]

    def fit_day(self, today, orders):
        pass

    def fit_bar(self, bar, orders):
        """
        止盈event： 较小利润值 < 买入后最大收益价格 - 当前价格 < 较大利润值
        :param bar: 当前驱动的交易时间金融时间序列数据
        :param orders: 买入择时策略中生成的订单序列
        :return:
        """
        for order in orders:
            # 通过order中的买入日期计算金融时间序列kl_pd中的index
            mask_time = self.kl_pd.index == order.buy_time
            start_ind = int(self.kl_pd[mask_time]['key'].values)
            end_ind = self.bar_ind + 1
            """
                从买入日子开始计算到今天得到买入后最大收盘价格作为max_close，
                注意如果是call找序列中的最大收盘价格，put找序列中的最小收盘价格
            """
            max_close = self.kl_pd.iloc[start_ind:end_ind, :].close.max() if order.buy_type_str == 'call' \
                else self.kl_pd.iloc[start_ind:end_ind, :].close.min()
            """
                max_close - order.buy_price * 方向 > today['atr21']：代表只针对有一定盈利的情况生效，即 > 较小利润值
                max_close - today.close * 方向 > today['atr21'] * self.close_atr_n：下跌了一定值止盈退出, 即 < 较大利润值
            """
            if (max_close - order.buy_price) * order.expect_direction > bar['atr21'] \
                    and (max_close - bar.close) * order.expect_direction > bar['atr21'] * self.close_atr_n:
                self.sell_next(order)
