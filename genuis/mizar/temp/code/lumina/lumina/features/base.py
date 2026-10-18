# -*- encoding:utf-8 -*-

class BuyFeatureMixin(object):
    """
        买入特征标识混入，与BuyUmpMixin不同，具体feature类可能属于多个类别
        即可能同时混入BuyFeatureMixin和SellFeatureMixin
    """
    _feature_buy = True
    _feature_buy_prefix = 'buy_'


class SellFeatureMixin(object):
    """
        卖出特征标识混入，与SellUmpMixin不同，具体feature类可能属于多个类别
        即可能同时混入BuyFeatureMixin和SellFeatureMixin
    """
    _feature_sell = True
    _feature_sell_prefix = 'sell_'


class FeatureBase(object):
    """特征构造基类"""

    def support_buy_feature(self):
        """是否支持买入特征构建"""
        return getattr(self, "_feature_buy", False) is True

    def support_sell_feature(self):
        """是否支持卖出特征构建"""
        return getattr(self, "_feature_sell", False) is True

    def check_support(self, buy_feature):
        """
        根据参数buy_feature检测是否支持特征构建
        :param buy_feature: 是否是买入特征构造（bool）
        """
        if buy_feature and not self.support_buy_feature:
            raise TypeError(
                'feature support buy must subclass BuyFeatureMixin!!!')
        if not buy_feature and not self.support_sell_feature:
            raise TypeError(
                'feature support buy must subclass SellFeatureMixin!!!')

    def feature_prefix(self, buy_feature, check=True):
        """
        根据buy_feature决定返回_feature_buy_prefix或者_feature_sell_prefix，目的是在calc_feature中构成唯一key
        :param buy_feature: 是否是买入特征构造（bool）
        :param check: 是否需要检测是否支持特征构建
        :return:
        """
        if check:
            self.check_support(buy_feature)

        return getattr(self,
                       '_feature_buy_prefix') if buy_feature else getattr(
                           self, '_feature_sell_prefix')

    def __str__(self):
        """打印对象显示：class name, support_buy_feature support_sell_feature, get_feature_keys"""
        return '{}:is_buy_feature:{} is_sell_feature:{} feature: {}'.format(
            self.__class__.__name__, self.support_buy_feature(),
            self.support_sell_feature(),
            self.get_feature_keys(self.support_buy_feature()))

    __repr__ = __str__

    def get_feature_ump_keys(self, ump_cls):
        """
        根据ump_cls，返回对应的get_feature_keys
        :param ump_cls: UltronUmpEdgeBase子类，参数为类，非实例对象
        :return: 键值对字典中的key序列
        """
        is_buy_ump = getattr(ump_cls, "_ump_type_prefix") == 'buy_'
        return self.get_feature_keys(buy_feature=is_buy_ump)

    def get_feature_keys(self, buy_feature):
        """
        子类主要需要实现的函数，定义feature的列名称
        :param buy_feature: 是否是买入特征构造（bool）
        :return: 键值对字典中的key序列
        """
        raise NotImplementedError('NotImplementedError get_feature_keys!!!')

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        """
        子类主要需要实现的函数，根据买入或者卖出时的金融时间序列，以及交易日信息构造特征
        :param kl_pd: 择时阶段金融时间序列
        :param combine_kl_pd: 合并择时阶段之前1年的金融时间序列
        :param day_ind: 交易发生的时间索引，即对应self.kl_pd.key
        :param buy_feature: 是否是买入特征构造（bool）
        :return: 构造特征的键值对字典
        """
        raise NotImplementedError('NotImplementedError calc_feature!!!')