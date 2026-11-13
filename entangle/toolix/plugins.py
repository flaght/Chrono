import importlib

def load_plugin_dynamically(plugin_names, codes):
    qubits = []
    for plugin_name in plugin_names:
        module_path = f"plugin.{plugin_name}.engine"
        try:
            module = importlib.import_module(module_path)
            EngineClass = getattr(module, "Engine")
            
            engine_instance = EngineClass(codes=codes)
            qubits.append(engine_instance)
            print(f"[OK] 插件 {plugin_name} 加载成功")
            
        except ModuleNotFoundError:
            print(f"[WARN] 未找到插件模块: {module_path}")
        except AttributeError:
            print(f"[WARN] 模块 {module_path} 中没有定义 Engine 类")
        except Exception as e:
            print(f"[ERROR] 初始化插件 {plugin_name} 时发生错误: {e}")
            
    return qubits


def load_monitor_plugin(main_engine, event_engine, account_configs,
                        gateway_module, storage_adapter=None,
                        **kwargs):
    """加载共享现有 MainEngine/Gateway 的统一监控插件。"""
    module = importlib.import_module("plugin.monitor.engine")
    engine_class = getattr(module, "Engine")
    return engine_class(
        main_engine=main_engine,
        event_engine=event_engine,
        account_configs=account_configs,
        gateway_module=gateway_module,
        storage_adapter=storage_adapter,
        **kwargs
    )
