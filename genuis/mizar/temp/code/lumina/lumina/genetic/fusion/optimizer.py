from typing import Dict, Tuple, Union
import pdb
import cvxpy as cp
import numpy as np
'''
使用协方差矩阵计算投资组合风险(cp.quad_form(x, self._variance))
设置目标波动率约束(risk <= self._variance_target)
包含权重总和等于1的约束(sum(x) == 1.0)
允许设置权重上下界约束
最小化风险或最大化风险调整后收益
均值方差优化需要同时考虑预期收益率和风险
当前使用L1范数(cp.pnorm(x - self._benchmark, 1))实现换手率约束是合理的
'''


class _TargetVarianceOptimizer(object):

    def __init__(
            self,
            objective: np.array,  ## 传入-objective, 资产预期收益率
            l1norm: float,  ## 换手率约束
            variance_target: float,  ## 目标波动率
            benchmark: np.array = None,  ## 基准组合权重
            rf: float = 0.0,  ## 无风险收益率
            variance: np.ndarray = None,  ## 资产协方差矩阵
            lookback_returns: np.ndarray = None,  ## 资产历史收益率
            lower_bound: Union[float, np.ndarray] = None,  ## 权重下界
            upper_bound: Union[float, np.ndarray] = None):  ## 权重上界

        self._n = len(objective)
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._l1norm = l1norm
        self._variance = variance
        self._variance_target = variance_target
        self._benchmark = benchmark if benchmark is not None else np.zeros(
            self._n)
        self._objective = objective
        self._lookback_returns = lookback_returns
        self._rf = rf

    def _prepare(self):
        x = cp.Variable(self._n)
        constraints = []
        if self._lower_bound is not None:
            constraints.append(x >= self._lower_bound)
        if self._upper_bound is not None:
            constraints.append(x <= self._upper_bound)

        if self._variance_target is not None:
            constraints.append(
                cp.quad_form(x, self._variance) <= self._variance_target**2)

        ## 换手率约束
        if self._l1norm is not None:
            constraints.append(
                cp.pnorm(x - self._benchmark, 1) <= self._l1norm)

        return x, constraints

    def _calculate_max_drawdown(self, returns: np.array) -> float:
        """计算最大回撤"""
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        return np.max(drawdown)

    def solver(self, solver: str = "ECOS", mode: str = "returns"):
        x, constraints = self._prepare()
        risk = cp.quad_form(x, self._variance)
        constraints.append(risk <= self._variance_target)
        constraints.append(cp.pnorm(x, 1) <= 1.0)
        constraints.append(sum(x) == 1.0)

        ## 最大化收益(在目标波动率约束下)
        if mode == 'returns':
            constraints.append(risk <= self._variance_target**2)
            prob = cp.Problem(cp.Minimize(x @ self._objective),
                              constraints=constraints)

        ## 最大化夏普比率
        elif mode == 'sharpe':
            prob = cp.Problem(cp.Minimize(
                (x @ self._objective - self._rf) / cp.sqrt(risk)),
                              constraints=constraints)

        ## 最大化卡玛
        elif mode == 'calmar':
            max_drawdown = self._calculate_max_drawdown(
                self._lookback_returns @ x.value if x.value is not None else 0)
            prob = cp.Problem(
                cp.Minimize((x @ self._objective) / max_drawdown), constraints)
        else:
            raise ValueError("Invalid mode: {}".format(mode))

        prob.solve(solver=solver)

        return x.value, prob.value, prob.status


class TargetVarianceOptimizer(object):

    def __init__(self,
                 objective: np.array,
                 current_pos: np.array,
                 target_turn_over: float,
                 target_vol: float,
                 cov: np.ndarray,
                 mode: str = "returns",
                 lbound: Union[float, np.ndarray] = None,
                 ubound: Union[float, np.ndarray] = None):
        self._optimizer = _TargetVarianceOptimizer(objective=objective,
                                                   benchmark=current_pos,
                                                   l1norm=target_turn_over,
                                                   variance_target=target_vol,
                                                   variance=cov,
                                                   lower_bound=lbound,
                                                   upper_bound=ubound)
        self._x, self._f_val, self._status = self._optimizer.solver(mode=mode)

    def status(self):
        return self._status

    def feval(self):
        return self._f_val

    def x_value(self):
        return self._x


def mean_variance_builder(er: np.array,
                          risk_model: Dict[str, Union[None, np.ndarray]],
                          turnover: float,
                          target_vol: float,
                          current_pos: np.array,
                          lbound: Union[float, np.ndarray] = None,
                          ubound: Union[float, np.ndarray] = None,
                          mode="returns") -> Tuple[str, float, np.array]:
    """
    mean_variance_builder
    :param er: 预期收益率
    :param risk_model: 风险模型
    :param turnover: 换手率
    :param target_vol: 目标波动率
    :param current_pos: 当前持仓
    :param lbound: 权重下界
    :param ubound: 权重上界
    :return:
    """
    optimizer = TargetVarianceOptimizer(objective=-er,
                                        current_pos=current_pos,
                                        target_turn_over=turnover,
                                        target_vol=target_vol,
                                        cov=risk_model,
                                        lbound=lbound,
                                        ubound=ubound,
                                        mode=mode)
    return optimizer.status(), optimizer.feval(), optimizer.x_value()
