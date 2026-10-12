"""本地无网络装配验证；仅替换不可用SDK实体，实际Runner/归属/风控/持久化不替换。"""
import ast, sys, types, abc
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from pathlib import Path
from importlib import import_module
from typing import Any, Callable, Mapping, Sequence
import unittest
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
for name,path in [('bomber.framework.market.stream','bomber/framework/market/stream'),('bomber.framework.trader','bomber/framework/trader'),
 ('bomber.framework.trader.runtime','bomber/framework/trader/runtime'),
 ('bomber.framework.trader.execution','bomber/framework/trader/execution'),
 ('bomber.framework.trader.execution.live','bomber/framework/trader/execution/live'),
 ('bomber.framework.trader.execution.simulation','bomber/framework/trader/execution/simulation'),
 ('bomber.framework.trader.execution.ctp','bomber/framework/trader/execution/ctp')]:
 m=types.ModuleType(name);m.__path__=[str(root/path)];sys.modules[name]=m
@dataclass(frozen=True)
class InstrumentId:
 value:str
 @property
 def symbol(self):return self.value.rsplit('.',1)[0]
 @classmethod
 def from_str(cls,value):return cls(value)
 def __str__(self):return self.value
class Price:
 def __init__(self,value): self.value=Decimal(str(value))
 def as_decimal(self):return self.value
 def __str__(self):return str(self.value)
class Bar:
 def __init__(self,**kw):self.__dict__.update(kw)
class BarType:
 def __init__(self,i,s):self.instrument_id=i;self.s=s
 @classmethod
 def from_str(cls,value):
  return cls(InstrumentId.from_str(value.split('-1-MINUTE-')[0]), value)
 def __str__(self):return self.s if self.s.endswith('-EXTERNAL') else f'{self.instrument_id}-{self.s}-LAST-EXTERNAL'
class DataType(Enum):
 BAR=1;CUSTOM_BAR=2;TRADE_TICK=3;QUOTE_TICK=4
base=types.ModuleType('bomber.framework.market.basic.base')
base.__dict__.update(dict(abc=abc,dataclass=dataclass,field=field,Decimal=Decimal,Any=Any,Callable=Callable,
 Mapping=Mapping,Sequence=Sequence,InstrumentId=InstrumentId,DataType=DataType,Bar=Bar,BarType=BarType,
 QuoteTick=type('QuoteTick',(),{}),TradeTick=type('TradeTick',(),{}),CustomBar=type('CustomBar',(),{})))
nodes=ast.parse((root/'bomber/framework/market/basic/base.py').read_text()).body
for name in ('InstrumentMeta','SubscriptionRequest','MarketDataFeed'):
 node=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==name)
 exec(compile(ast.Module([node],[]),'<actual SDK registration contract>','exec'),base.__dict__)
def make_bar(*,instrument_id,meta=None,bar_type='1-MINUTE',**kw):
 for k in ('open','high','low','close','volume'):
  v=kw[k];precision=meta.size_precision if k=='volume' else meta.price_precision
  kw[k]=Price(Decimal(str(v)).quantize(Decimal(1).scaleb(-precision)))
 return Bar(bar_type=bar_type if isinstance(bar_type,BarType) else BarType(instrument_id,bar_type),**kw)
base.make_bar=make_bar
# SDK construction entry points imported by the real aggregator; unused for quote accumulation.
base.make_trade_tick=lambda **kw: types.SimpleNamespace(**kw)
sys.modules[base.__name__]=base
# These unused native SDK bridges are never instantiated in Model/Recording tests.
gateway=types.ModuleType('bomber.framework.trader.execution.simulation.gateway')
gateway.NautilusOrderGateway=type('NautilusOrderGateway',(),{});sys.modules[gateway.__name__]=gateway
profile=types.ModuleType('bomber.framework.trader.execution.ctp.profile');sys.modules[profile.__name__]=profile
profile.__dict__.update(dict(dataclass=dataclass,Decimal=Decimal,PositionEffect=import_module('bomber.framework.trader.execution.contracts').PositionEffect))
node_list=ast.parse((root/'bomber/framework/trader/execution/ctp/profile.py').read_text()).body
for node in node_list:
 if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in ('_decimal','CtpCommissionRule'):
  exec(compile(ast.Module([node],[]),'<actual commission rules>','exec'),profile.__dict__)
ctp=sys.modules['bomber.framework.trader.execution.ctp']
for name,module in [('CtpPositionLedger','ledger'),('CtpLimitPlanner','limit_planner'),('CtpExecutionAccounting','accounting')]:
 setattr(ctp,name,getattr(import_module('bomber.framework.trader.execution.ctp.'+module),name))
live=sys.modules['bomber.framework.trader.execution.live']
for name,module in [('ControlledLiveExecutionClient','controlled'),('NautilusLiveExecutionBackend','backend')]:
 setattr(live,name,getattr(import_module('bomber.framework.trader.execution.live.'+module),name))
sim=sys.modules['bomber.framework.trader.execution.simulation'];sim.SimulationExecutionClient=import_module(sim.__name__+'.client').SimulationExecutionClient
trader=sys.modules['bomber.framework.trader'];trader.RuntimeMode=import_module(trader.__name__+'.contracts').RuntimeMode
trader.UnifiedStrategyRunner=import_module(trader.__name__+'.runner').UnifiedStrategyRunner
# Attribution regression uses the same actual public modules and SDK entity shim.
exports = {
 'contracts': ('ExecutionRoute',),
 'dynamic_routes': ('ContractAssignment', 'DynamicExecutionRoute', 'ScheduledContractResolver'),
 'portfolio': ('PositionManager',),
 'persistence': ('JsonStateStore', 'RuntimeStateManager'),
 'template': ('StrategyTemplate',),
 'execution.live.client': ('BackendExecutionClient',),
 'execution.contracts': ('ExecutionReport', 'ExecutionReportType'),
 'execution.planner': ('NetTargetOrderPlanner',),
}
for suffix, names in exports.items():
 module = import_module(trader.__name__ + '.' + suffix)
 for name in names:
  setattr(trader, name, getattr(module, name))
suite=unittest.defaultTestLoader.loadTestsFromNames([
 'tests.test_option_vega_assembly', 'tests.test_position_attribution', 'tests.test_index_query_feed', 'tests.test_option_replay_test', 'tests.test_option_execution_audit'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(not result.wasSuccessful())
