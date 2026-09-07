"""Polars implementation of the time-series factor evaluator.

The calculation path stays in Polars.  Pandas is materialized only when the
legacy plotting/saving API is used.
"""

import os
from xml.dom import minidom
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl


def _rolling_minimum_samples(value):
    """Support the Polars 1.21 min_periods -> min_samples rename."""
    parts = pl.__version__.split('.')
    try:
        version = tuple(int(part.split('-')[0]) for part in parts[:2])
    except ValueError:
        version = (1, 21)
    key = 'min_samples' if version >= (1, 21) else 'min_periods'
    return {key: value}


class FactorEvaluate1:
    """High-performance, API-compatible time-series factor evaluator."""

    def __init__(self,
                 factor_data,
                 resampling_win: int = 1,
                 factor_name: str = 'factor',
                 ret_name: str = 'ret',
                 roll_win: int = 252,
                 fee: float = 0.0003,
                 scale_method: str = 'roll_min_max',
                 annualization_factor: int = 252,
                 expression=None,
                 name=None):
        self.factor_name = factor_name
        self.ret_name = ret_name
        self.roll_win = int(roll_win)
        self.fee = float(fee)
        self.scale_method = scale_method
        self.annualization_factor = int(annualization_factor)
        self.name = name
        self.expression = expression
        self.stats = None
        self.resampling_win = int(resampling_win)
        self.figure = None
        self.direction_inverted = False
        self.factor_data = self._preprocess_data(factor_data)
        self.resample_data_pl = None
        self.resample_data = None

    def _preprocess_data(self, data):
        if isinstance(data, pd.DataFrame):
            data = pl.from_pandas(data)
        elif isinstance(data, pl.LazyFrame):
            data = data.collect()
        elif not isinstance(data, pl.DataFrame):
            raise TypeError('factor_data must be pandas/polars DataFrame or Polars LazyFrame')

        required = {'trade_time', self.factor_name, self.ret_name}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(f'Missing required columns: {missing}')

        time_dtype = data.schema['trade_time']
        if time_dtype == pl.String:
            time_expr = pl.col('trade_time').str.to_datetime(strict=False)
        elif time_dtype == pl.Date:
            time_expr = pl.col('trade_time').cast(pl.Datetime)
        else:
            time_expr = pl.col('trade_time').cast(pl.Datetime, strict=False)

        return (
            data.lazy()
            .select(
                time_expr.alias('trade_time'),
                pl.col(self.factor_name).cast(pl.Float64, strict=False),
                pl.col(self.ret_name).cast(pl.Float64, strict=False),
            )
            .with_columns(
                pl.when(pl.col(self.factor_name).is_finite())
                .then(pl.col(self.factor_name)).otherwise(None)
                .alias(self.factor_name),
                pl.when(pl.col(self.ret_name).is_finite())
                .then(pl.col(self.ret_name)).otherwise(None)
                .alias(self.ret_name),
            )
            .drop_nulls('trade_time')
            .sort('trade_time')
            .collect()
        )

    def _scaled_frame(self):
        x = pl.col(self.factor_name)
        win = self.roll_win

        if self.scale_method == 'roll_min_max':
            lo = x.rolling_min(window_size=win)
            hi = x.rolling_max(window_size=win)
            denominator = (hi - lo).clip(lower_bound=1e-8)
            scaled = 2 * (x - lo) / denominator - 1
        elif self.scale_method == 'roll_zscore':
            mean = x.rolling_mean(window_size=win)
            std = x.rolling_std(window_size=win).clip(lower_bound=1e-8)
            scaled = ((x - mean) / std).clip(-3, 3) / 3
        elif self.scale_method == 'roll_quantile':
            q25 = x.rolling_quantile(
                0.25, interpolation='linear', window_size=win)
            q75 = x.rolling_quantile(
                0.75, interpolation='linear', window_size=win)
            scaled = 2 * (x - q25) / (q75 - q25).clip(lower_bound=1e-8) - 1
        elif self.scale_method == 'ew_zscore':
            mean = x.ewm_mean(span=win, adjust=False)
            variance = x.ewm_var(span=win, adjust=False)
            scaled = ((x - mean) / variance.sqrt().clip(lower_bound=1e-8)).clip(-3, 3) / 3
        elif self.scale_method == 'train_const':
            training = self.factor_data.get_column(self.factor_name).head(win)
            mean = training.mean()
            std = training.std()
            std = max(std, 1e-8) if std is not None and np.isfinite(std) else 1e-8
            scaled = ((x - mean) / std).clip(-3, 3) / 3
        elif self.scale_method == 'raw':
            scaled = x
        else:
            raise ValueError(f'Unknown scale_method: {self.scale_method}')

        return self.factor_data.with_columns(scaled.alias('f_scaled'))

    @staticmethod
    def _item(frame, expression):
        return frame.select(expression).item()

    @staticmethod
    def _safe_ratio(numerator, denominator, default=0.0):
        if denominator is None or not np.isfinite(denominator) or denominator == 0:
            return default
        if numerator is None or not np.isfinite(numerator):
            return default
        return numerator / denominator

    def run(self, is_check=False):
        if self.resampling_win <= 0:
            raise ValueError('resampling_win must be a positive integer')

        frame = (
            self._scaled_frame().lazy()
            .filter(pl.col('trade_time').dt.minute() % self.resampling_win == 0)
            .drop_nulls(self.ret_name)
            .with_columns(
                pl.when(pl.col('f_scaled').is_finite())
                .then(pl.col('f_scaled')).otherwise(None)
                .alias('f_scaled'))
            .collect()
        )
        if frame.is_empty():
            raise ValueError('No valid return observations after resampling.')

        # Original/raw time-series IC is retained; effective IC is reported
        # separately after the direction adjustment.
        frame = frame.with_columns(
            pl.rolling_corr(
                pl.col(self.ret_name),
                pl.col(self.factor_name),
                window_size=self.roll_win,
                **_rolling_minimum_samples(5),
            ).alias('ic')
        ).with_columns(pl.col('ic').cum_sum().alias('cumsum_ic'))

        ic_values = frame.select(
            pl.corr(self.ret_name, self.factor_name).alias('total_ic'),
            pl.col('ic').mean().alias('ic_mean'),
            pl.col('ic').std().alias('ic_std'),
        ).row(0, named=True)
        total_ic = (ic_values['total_ic']
                    if ic_values['total_ic'] is not None else np.nan)
        ic_mean = (ic_values['ic_mean']
                   if ic_values['ic_mean'] is not None else np.nan)
        ic_std = (ic_values['ic_std']
                  if ic_values['ic_std'] is not None else np.nan)
        ic_ir = self._safe_ratio(ic_mean, ic_std)
        self.direction_inverted = bool(
            np.isfinite(ic_mean) and ic_mean < 0)
        direction = -1.0 if self.direction_inverted else 1.0

        if frame.get_column('f_scaled').drop_nulls().is_empty():
            self.stats = self._invalid_stats(
                direction_inverted=self.direction_inverted)
            self.resample_data_pl = frame
            self.resample_data = None
            return self.stats

        frame = (
            frame.lazy()
            .with_columns((pl.col('f_scaled') * direction).alias('f_scaled'))
            .with_columns(pl.col('f_scaled').fill_null(0.0).alias('pos'))
            .with_columns(
                (pl.col('pos') * pl.col(self.ret_name)).alias('gross_ret'),
                pl.col('pos').diff().fill_null(pl.col('pos').abs()).abs().alias('turnover'),
            )
            .with_columns(
                (pl.col('gross_ret') - self.fee * pl.col('turnover')).alias('net_ret'))
            .with_columns((pl.lit(1.0) + pl.col('net_ret')).cum_prod().alias('nav'))
            .collect()
        )

        daily = (
            frame.lazy()
            .with_columns(pl.col('trade_time').dt.date().alias('_date'))
            .group_by('_date')
            .agg(((pl.col('net_ret') + 1).product() - 1).alias('daily_ret'))
            .sort('_date')
            .collect()
        )
        daily_mean = daily.get_column('daily_ret').mean()
        daily_std = daily.get_column('daily_ret').std()
        annualized_sharpe = self._safe_ratio(
            daily_mean * self.annualization_factor,
            daily_std * np.sqrt(self.annualization_factor))

        nav = frame.get_column('nav')
        total_ret = nav[-1] - 1
        elapsed_days = (frame['trade_time'][-1] - frame['trade_time'][0]).days
        years = max(elapsed_days / 365, 1e-6)
        annualized_return = (1 + total_ret)**(1 / years) - 1
        max_dd = self._item(
            frame,
            (pl.col('nav') / pl.col('nav').cum_max().clip(lower_bound=1.0) - 1).min())

        winning = frame.filter(pl.col('net_ret') > 0).get_column('net_ret')
        losing = frame.filter(pl.col('net_ret') < 0).get_column('net_ret')
        if winning.is_empty():
            profit_ratio = 0.0
        elif losing.is_empty():
            profit_ratio = np.inf
        else:
            profit_ratio = winning.mean() / abs(losing.mean())

        factor = pl.col(self.factor_name)
        ret = pl.col(self.ret_name)
        distribution = frame.select(
            factor.is_not_null().mean().alias('factor_coverage'),
            pl.col('f_scaled').is_not_null().mean().alias('signal_coverage'),
            factor.mean().alias('factor_mean'),
            factor.std().alias('factor_std'),
            factor.skew(bias=False).alias('factor_skew'),
            factor.kurtosis(fisher=True, bias=False).alias('factor_kurtosis'),
            pl.corr(factor, factor.shift(1)).alias('factor_autocorr'),
            pl.corr(ret, ret.shift(1)).alias('ret_autocorr'),
        ).row(0, named=True)
        distribution = {
            key: value if value is not None else np.nan
            for key, value in distribution.items()
        }
        period_mean = frame.get_column('net_ret').mean()
        period_std = frame.get_column('net_ret').std()

        self.stats = {
            'total_ret': total_ret,
            'avg_ret': period_mean,
            'max_dd': max_dd,
            'calmar': annualized_return / abs(max_dd) if max_dd else np.nan,
            'sharpe1': self._safe_ratio(period_mean, period_std),
            'sharpe2': annualized_sharpe,
            'turnover': frame.get_column('turnover').mean(),
            'win_rate': self._item(frame, (pl.col('net_ret') > 0).mean()),
            'profit_ratio': profit_ratio,
            'total_ic': total_ic,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            **distribution,
            'direction_inverted': self.direction_inverted,
            'effective_total_ic': total_ic * direction,
            'effective_ic_mean': ic_mean * direction,
            'effective_ic_std': ic_std,
            'effective_ic_ir': ic_ir * direction,
        }
        self.resample_data_pl = frame
        self.resample_data = None
        if is_check:
            self._check_warnings()
        return self.stats

    def _invalid_stats(self, direction_inverted=False):
        return {
            'total_ret': -1.0, 'avg_ret': -1.0, 'max_dd': np.nan,
            'calmar': -10.0, 'sharpe1': -1.0, 'sharpe2': -10.0,
            'turnover': 10.0, 'win_rate': 0.0, 'profit_ratio': 0.0,
            'total_ic': 0.0, 'ic_mean': 0.0, 'ic_std': 1.0, 'ic_ir': 1.0,
            'factor_autocorr': np.nan, 'ret_autocorr': np.nan,
            'factor_coverage': 0.0, 'factor_mean': np.nan,
            'signal_coverage': 0.0,
            'factor_std': np.nan, 'factor_skew': np.nan,
            'factor_kurtosis': np.nan,
            'direction_inverted': direction_inverted,
            'effective_total_ic': np.nan, 'effective_ic_mean': np.nan,
            'effective_ic_std': np.nan, 'effective_ic_ir': np.nan,
        }

    def _check_warnings(self):
        """Print lightweight sanity checks without depending on the pandas version."""
        print("\n--- Sanity Checks & Warnings ---")
        ret_ac = self.stats.get('ret_autocorr', np.nan)
        if not np.isfinite(ret_ac) or not (-0.1 < ret_ac < 0.1):
            print(f"WARNING: Return autocorrelation is {ret_ac:.3f}.")
        else:
            print(f"Return autocorrelation ({ret_ac:.3f}) is normal.")

        factor_ac = self.stats.get('factor_autocorr', np.nan)
        if np.isfinite(factor_ac) and abs(factor_ac) > 0.99:
            print(f"WARNING: Factor autocorrelation is {factor_ac:.3f}.")
        else:
            print(f"Factor autocorrelation ({factor_ac:.3f}) is within range.")

        ic_ir = self.stats.get('ic_ir', np.nan)
        if not np.isfinite(ic_ir) or ic_ir < 0.3:
            print(f"WARNING: ICIR is {ic_ir:.3f}.")
        else:
            print(f"ICIR ({ic_ir:.3f}) indicates stable performance.")

    def _generate_stats_text(self):
        parts = []
        if self.expression is not None:
            parts.append(f"Expression: {self.expression}")
        if self.name is not None:
            parts.append(f"Name: {self.name}")
        if parts:
            parts.append('')
        parts.append(
            "--- Performance Metrics ---\n"
            f"{'Avg Return':<20}: {self.stats['avg_ret'] * 10000:.2f} bps\n"
            f"{'Total Return':<20}: {self.stats['total_ret']:.2%}\n"
            f"{'Sharpe Ratio':<20}: {self.stats['sharpe1']:.2f}\n"
            f"{'Ann Sharpe Ratio':<20}: {self.stats['sharpe2']:.2f}\n"
            f"{'Max Drawdown':<20}: {self.stats['max_dd']:.2%}\n"
            f"{'Calmar Ratio':<20}: {self.stats['calmar']:.2f}\n"
            f"{'Win Rate':<20}: {self.stats['win_rate']:.2%}\n"
            f"{'Profit/Loss Ratio':<20}: {self.stats['profit_ratio']:.2f}\n"
            "\n--- Factor Characteristics ---\n"
            f"{'Total IC':<20}: {self.stats['total_ic']:.4f}\n"
            f"{'IC Mean':<20}: {self.stats['ic_mean']:.4f}\n"
            f"{'ICIR':<20}: {self.stats['ic_ir']:.4f}\n"
            f"{'Mean Turnover':<20}: {self.stats['turnover']:.4f}\n"
            f"{'Factor Coverage':<20}: {self.stats['factor_coverage']:.2%}\n"
            f"{'Signal Coverage':<20}: {self.stats['signal_coverage']:.2%}\n"
            f"{'Factor Autocorr':<20}: {self.stats['factor_autocorr']:.4f}\n"
            f"{'Return Autocorr':<20}: {self.stats['ret_autocorr']:.4f}\n"
            f"{'Roll Window':<20}: {self.roll_win}\n"
            f"{'Resampling Window':<20}: {self.resampling_win}\n")
        return '\n'.join(parts)

    def to_pandas(self):
        """Materialize the evaluated time series for legacy consumers."""
        if self.resample_data_pl is None:
            raise RuntimeError("Please run the 'run()' method first.")
        if self.resample_data is None:
            self.resample_data = self.resample_data_pl.to_pandas().set_index('trade_time')
        return self.resample_data

    def plot_results(self):
        if self.stats is None:
            raise RuntimeError("Please run the 'run()' method before plotting.")
        data = self.to_pandas()
        fig, axes = plt.subplots(3, 2, figsize=(18, 16))
        fig.suptitle(
            f"Factor Evaluation: {self.factor_name} vs {self.ret_name} | "
            f"roll_win={self.roll_win}, resampling_win={self.resampling_win}, "
            f"scale_method={self.scale_method}", fontsize=16)

        data['nav'].plot(ax=axes[0, 0], label='Net Asset Value (NAV)', color='blue')
        (1 + data['gross_ret']).cumprod().plot(
            ax=axes[0, 0], label='Cumulative Gross Return',
            color='orange', linestyle='--')
        axes[0, 0].set_title('Performance')
        axes[0, 0].set_ylabel('NAV')
        axes[0, 0].set_xlabel('trade_time (sequential)')
        axes[0, 0].legend()

        axes[0, 1].axis('off')
        axes[0, 1].text(0.03, 0.97, self._generate_stats_text(),
                        va='top', family='monospace', fontsize=11)
        axes[0, 1].set_title('Key Performance Indicators')

        ic_ax = axes[1, 0]
        data['ic'].plot(ax=ic_ax, label='Rolling IC', color='steelblue',
                        alpha=0.8)
        cumulative_ic_ax = ic_ax.twinx()
        data['cumsum_ic'].plot(ax=cumulative_ic_ax, label='Cumulative IC',
                               color='black', linestyle='--', linewidth=1.5)
        ic_ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ic_ax.set_title('IC Analysis')
        ic_ax.set_xlabel('trade_time (sequential)')
        ic_ax.set_ylabel('Rolling IC', color='steelblue')
        cumulative_ic_ax.set_ylabel('Cumulative IC', color='black')
        lines = ic_ax.get_lines()[:1] + cumulative_ic_ax.get_lines()
        ic_ax.legend(lines, [line.get_label() for line in lines], loc='upper left')

        axes[1, 1].scatter(data[self.factor_name], data[self.ret_name],
                           s=10, alpha=0.3, color='purple')
        axes[1, 1].set_title('Factor vs. Return Scatter Plot')
        axes[1, 1].set_xlabel('Original Factor Value')
        axes[1, 1].set_ylabel('Forward Return')

        drawdown = (data['nav'] /
                    data['nav'].cummax().clip(lower=1.0) - 1) * 100
        drawdown.plot(ax=axes[2, 0], color='red')
        axes[2, 0].fill_between(drawdown.index, drawdown.values, 0,
                                color='red', alpha=0.2)
        axes[2, 0].set_title(
            f"Drawdown Over Time (Max = {self.stats['max_dd']:.2%})")
        axes[2, 0].set_ylabel('Drawdown (%)')
        axes[2, 0].set_xlabel('trade_time (sequential)')

        data['turnover'].plot(ax=axes[2, 1], color='teal')
        axes[2, 1].set_title(
            f"Turnover Over Time (Mean = {self.stats['turnover']:.3f})")
        axes[2, 1].set_ylabel('Turnover')
        axes[2, 1].set_xlabel('trade_time (sequential)')
        for ax in axes.flat:
            ax.grid(True)
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        self.figure = fig
        return fig

    def generate_xml(self, start_time, end_time):
        if self.stats is None:
            raise RuntimeError("Please run the 'run()' method first.")

        def add_text(parent, tag, value, **attributes):
            element = ET.SubElement(parent, tag, {
                key: str(item) for key, item in attributes.items()
                if item is not None
            })
            if value is None:
                element.set('status', 'missing')
            else:
                element.text = str(value)
            return element

        def add_metric(parent, tag, value, description, unit='ratio'):
            element = ET.SubElement(parent, tag, {
                'description': description, 'unit': unit
            })
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = np.nan
            if np.isfinite(number):
                element.text = format(number, '.12g')
            else:
                element.set('status', 'missing')
            return element

        root = ET.Element('factor_evaluation', {
            'schema_version': '1.0.0', 'purpose': 'factor_screening'
        })
        identity = ET.SubElement(root, 'identity')
        add_text(identity, 'name', self.name)
        add_text(identity, 'expression', self.expression)
        add_text(identity, 'factor_field', self.factor_name)
        add_text(identity, 'forward_return_field', self.ret_name)

        config = ET.SubElement(root, 'evaluation_config')
        add_text(config, 'start_time', start_time)
        add_text(config, 'end_time', end_time)
        add_text(config, 'rolling_window', self.roll_win, unit='period')
        add_text(config, 'resampling_window', self.resampling_win, unit='minute')
        add_text(config, 'scale_method', self.scale_method)
        add_text(config, 'annualization_factor', self.annualization_factor,
                 unit='trading_days_per_year')
        add_text(config, 'return_basis', 'net_return_after_costs')
        add_text(config, 'engine', 'polars')

        adjustment = ET.SubElement(root, 'direction_adjustment')
        add_text(adjustment, 'applied',
                 str(self.stats.get('direction_inverted', False)).lower(),
                 unit='boolean')
        add_text(adjustment, 'rule',
                 'invert f_scaled when pre-adjustment rolling IC mean is negative')

        metrics = ET.SubElement(root, 'metrics', {
            'basis': 'net_return_after_costs'
        })
        for tag, key, description, unit in (
            ('average_return', 'avg_ret', '平均收益', 'decimal_return'),
            ('total_return', 'total_ret', '累计成本后收益', 'decimal_return'),
            ('annualized_sharpe', 'sharpe2', '年化夏普', 'ratio'),
            ('maximum_drawdown', 'max_dd', '最大回撤', 'decimal_return'),
            ('calmar_ratio', 'calmar', '卡玛', 'ratio'),
            ('win_rate', 'win_rate', '胜率', 'decimal_ratio'),
            ('profit_loss_ratio', 'profit_ratio', '盈亏比', 'ratio'),
            ('turnover', 'turnover', '平均换手率', 'turnover'),
        ):
            add_metric(metrics, tag, self.stats.get(key), description, unit)

        characteristics = ET.SubElement(root, 'factor_characteristics', {
            'ic_scope': 'time_series'
        })
        raw_ic = ET.SubElement(characteristics, 'raw_ic', {
            'description': '方向调整前的原始因子IC'
        })
        effective_ic = ET.SubElement(characteristics, 'effective_ic', {
            'description': '方向调整后实际用于收益计算的因子IC'
        })
        for parent, prefix in ((raw_ic, ''), (effective_ic, 'effective_')):
            add_metric(parent, 'total_ic', self.stats.get(f'{prefix}total_ic'),
                       '全样本IC', 'correlation')
            add_metric(parent, 'ic_mean', self.stats.get(f'{prefix}ic_mean'),
                       '滚动IC平均值', 'correlation')
            add_metric(parent, 'ic_std', self.stats.get(f'{prefix}ic_std'),
                       '滚动IC标准差', 'correlation')
            add_metric(parent, 'ic_ir', self.stats.get(f'{prefix}ic_ir'),
                       '滚动ICIR', 'ratio')
        add_metric(characteristics, 'factor_autocorrelation',
                   self.stats.get('factor_autocorr'), '原始因子一阶自相关',
                   'correlation')
        add_metric(characteristics, 'return_autocorrelation',
                   self.stats.get('ret_autocorr'), '目标收益一阶自相关',
                   'correlation')

        distribution = ET.SubElement(root, 'factor_distribution')
        for tag, key, description, unit in (
            ('coverage', 'factor_coverage', '有效因子值覆盖率', 'decimal_ratio'),
            ('signal_coverage', 'signal_coverage',
             '标准化后有效交易信号覆盖率', 'decimal_ratio'),
            ('mean', 'factor_mean', '原始因子均值', 'factor_value'),
            ('std', 'factor_std', '原始因子标准差', 'factor_value'),
            ('skew', 'factor_skew', '原始因子偏度', 'ratio'),
            ('kurtosis', 'factor_kurtosis', '原始因子超额峰度', 'ratio'),
        ):
            add_metric(distribution, tag, self.stats.get(key), description, unit)

        rough_xml = ET.tostring(root, encoding='utf-8')
        return minidom.parseString(rough_xml).toprettyxml(
            indent='  ', encoding='utf-8').decode('utf-8')

    def save_results(self, base_output_dir: str):
        if self.stats is None:
            raise RuntimeError("Please run the 'run()' method before saving.")
        output_dir = os.path.join(base_output_dir, str(self.name))
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'performance_summary.txt'),
                  'w', encoding='utf-8') as file:
            file.write(self._generate_stats_text())

        data = self.to_pandas()
        for metric in ('nav', 'ic', 'turnover'):
            if metric in data.columns:
                data[metric].to_csv(os.path.join(output_dir, f'{metric}.csv'),
                                    header=True)

        if self.figure is not None:
            self.figure.savefig(os.path.join(output_dir, 'evaluation_plot.png'),
                                dpi=150)
            plot_dir = os.path.join(base_output_dir, 'plot')
            os.makedirs(plot_dir, exist_ok=True)
            self.figure.savefig(os.path.join(plot_dir, f'{self.name}.png'), dpi=150)

        xml_text = self.generate_xml(
            self.resample_data_pl['trade_time'][0].isoformat(),
            self.resample_data_pl['trade_time'][-1].isoformat())
        with open(os.path.join(output_dir, 'evaluation.xml'),
                  'w', encoding='utf-8') as file:
            file.write(xml_text)
        xml_dir = os.path.join(base_output_dir, 'xml')
        os.makedirs(xml_dir, exist_ok=True)
        with open(os.path.join(xml_dir, f'{self.name}.xml'),
                  'w', encoding='utf-8') as file:
            file.write(xml_text)

    def cal_returns(self):
        if self.resample_data_pl is None or 'net_ret' not in self.resample_data_pl.columns:
            raise RuntimeError("Please run the 'run()' method first.")
        frame = self.resample_data_pl
        result = {}
        for label, predicate in (
                ('long', pl.col('f_scaled') > 0),
                ('short', pl.col('f_scaled') < 0)):
            values = frame.filter(predicate).get_column('net_ret')
            result[f'{label}_count'] = len(values)
            result[f'{label}_sum_returns'] = values.sum() if len(values) else 0.0
            result[f'{label}_avg_returns'] = values.mean() if len(values) else 0.0
            result[f'{label}_win_ratio'] = ((values > 0).sum() / len(values)
                                             if len(values) else 0.0)
        return result


__all__ = ['FactorEvaluate1']
