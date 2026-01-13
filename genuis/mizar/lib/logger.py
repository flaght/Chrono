# rich_logger.py
import logging
import pandas as pd
from typing import Optional, Iterable
from io import StringIO

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.progress import track

class RichLogger:
    """
    一个集成了Rich库美化功能和标准logging模块的日志记录器类。

    该类提供了一个统一的接口，用于记录事件性日志（如info, warning）
    和打印结构化的富文本内容（如标题、面板、表格）。
    """

    def __init__(self, name: str, verbose: bool = True, log_file: Optional[str] = None):
        """
        初始化RichLogger。

        Args:
            name (str): 日志记录器的名称。
            verbose (bool, optional): 是否开启详细模式。
                - True: 显示INFO及以上级别的日志。
                - False: 仅显示WARNING及以上级别的日志。
                默认为 True。
            log_file (Optional[str], optional): 可选的日志文件名。如果提供，
                日志也会以纯文本格式写入此文件。默认为 None。
        """
        self.verbose = verbose
        self.log_file = log_file
        self.console = Console()
        self.logger = logging.getLogger(name)
        
        self._setup_logger()

    def _setup_logger(self):
        """内部方法，根据当前设置配置logger handlers。"""
        # 清空已有handlers，以便动态切换级别
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # 设置logger的基础级别为DEBUG，让所有信息都能通过logger，过滤在handler层完成
        self.logger.setLevel(logging.DEBUG)

        # 根据 verbose 参数决定handler的日志级别
        log_level = logging.INFO if self.verbose else logging.WARNING

        # 1. 配置 RichHandler (用于控制台输出)
        rich_handler = RichHandler(
            console=self.console,
            show_path=False,
            markup=True,
            rich_tracebacks=True # 美化异常堆栈信息
        )
        rich_handler.setLevel(log_level)
        self.logger.addHandler(rich_handler)

        # 2. 如果提供了文件名，配置 FileHandler (用于文件输出)
        # 文件日志始终记录所有级别（DEBUG及以上），不受verbose参数影响
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别的日志
            # 文件日志使用标准格式，因为rich标记在文件中无效
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _write_to_file(self, text: str):
        """将文本内容写入日志文件（如果已设置）。"""
        if self.log_file:
            # 直接写入文件，不通过logger（避免重复格式化）
            # 允许空行写入（用于spacer方法）
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    # 如果text为空字符串，只写入换行符
                    if text:
                        f.write(text.rstrip() + '\n')
                    else:
                        f.write('\n')
            except Exception as e:
                # 如果写入失败，通过logger记录错误
                self.logger.error(f"写入日志文件失败: {e}")
    
    def _capture_rich_text(self, *objects, **kwargs) -> str:
        """捕获Rich对象的纯文本输出。"""
        # 创建一个临时的StringIO来捕获输出
        string_io = StringIO()
        temp_console = Console(file=string_io, width=120, force_terminal=False)
        temp_console.print(*objects, **kwargs)
        return string_io.getvalue()

    def set_verbose(self, verbose: bool):
        """
        动态设置日志的详细模式。

        Args:
            verbose (bool): True为详细模式（INFO级别），False为标准模式（WARNING级别）。
        """
        if self.verbose == verbose:
            return # 状态未改变，无需操作
        
        self.verbose = verbose
        self._setup_logger()
        self.info(f"日志模式已切换为: {'[bold green]详细[/bold green]' if verbose else '[bold yellow]标准[/bold yellow]'}")

    def set_level(self, level: int):
        """
        动态设置日志级别。

        Args:
            level (int): 日志级别，使用 logging 模块的常量：
                - logging.DEBUG (10): 显示所有日志
                - logging.INFO (20): 显示INFO及以上级别
                - logging.WARNING (30): 显示WARNING及以上级别
                - logging.ERROR (40): 仅显示ERROR及以上级别
                - logging.CRITICAL (50): 仅显示CRITICAL级别

        示例:
            from lib.utils.logger import logger
            import logging
            logger.set_level(logging.DEBUG)  # 显示所有日志
            logger.set_level(logging.WARNING)  # 仅显示警告和错误
        """
        # 将日志级别映射到verbose模式
        if level <= logging.DEBUG:
            new_verbose = True
        elif level <= logging.INFO:
            new_verbose = True
        elif level <= logging.WARNING:
            new_verbose = False
        else:
            new_verbose = False
        
        # 如果verbose模式需要改变，则更新
        if self.verbose != new_verbose:
            self.verbose = new_verbose
        
        # 重新设置handler的级别
        self._setup_logger()
        
        # 更新RichHandler的级别
        for handler in self.logger.handlers:
            if isinstance(handler, RichHandler):
                handler.setLevel(level)
        
        level_name = logging.getLevelName(level)
        self.info(f"日志级别已设置为: [bold cyan]{level_name}[/bold cyan] ({level})")

    def set_log_file(self, log_file: Optional[str]):
        """
        动态设置日志文件路径。

        Args:
            log_file (Optional[str]): 日志文件路径。如果为 None，则禁用文件日志。
                如果提供路径，日志会以追加模式写入该文件。

        示例:
            from lib.utils.logger import logger
            logger.set_log_file("new_log.log")  # 设置新的日志文件
            logger.set_log_file(None)  # 禁用文件日志
        """
        if self.log_file == log_file:
            return # 状态未改变，无需操作
        
        self.log_file = log_file
        self._setup_logger()
        if log_file:
            self.info(f"日志文件已设置为: [bold cyan]{log_file}[/bold cyan]")
        else:
            self.info("文件日志已禁用")
    
    def configure(self, verbose: Optional[bool] = None, log_file: Optional[str] = None, level: Optional[int] = None):
        """
        一次性配置多个日志参数。

        Args:
            verbose (Optional[bool]): 是否开启详细模式。如果提供，将覆盖level参数。
            log_file (Optional[str]): 日志文件路径。None表示禁用文件日志。
            level (Optional[int]): 日志级别（logging.DEBUG, logging.INFO等）。
                如果同时提供verbose和level，verbose优先。

        示例:
            from lib.utils.logger import logger
            import logging
            
            # 同时设置多个参数
            logger.configure(verbose=True, log_file="new_log.log")
            
            # 或者使用日志级别
            logger.configure(level=logging.DEBUG, log_file="debug.log")
        """
        if verbose is not None:
            self.set_verbose(verbose)
        elif level is not None:
            self.set_level(level)
        
        if log_file is not None:
            self.set_log_file(log_file)

    # --- 标准日志方法 ---
    def info(self, message: str, *args, **kwargs):
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs):
        self.logger.debug(message, *args, **kwargs)

    # --- Rich 组件便捷方法 ---
    def rule(self, title: str, style: str = "bold yellow"):
        """打印一个带标题的水平分割线。"""
        rule_obj = Rule(f"[{style}]{title}[/{style}]", style=style)
        self.console.print(rule_obj)
        # 同时写入文件
        text = self._capture_rich_text(rule_obj)
        self._write_to_file(text)

    def panel(self, content, title: str, border_style: str = "green"):
        """将内容包裹在一个带标题的面板中打印。"""
        panel_obj = Panel(
            content,
            title=f"[bold]{title}[/bold]",
            title_align="left",
            border_style=border_style
        )
        self.console.print(panel_obj)
        # 同时写入文件
        text = self._capture_rich_text(panel_obj)
        self._write_to_file(text)

    def table(self, data, title: str):
        """
        将 Pandas DataFrame 或 Series 打印成一个美观的表格。
        """
        # --- 新增的逻辑：检查输入类型 ---
        if isinstance(data, pd.Series):
            # 如果是 Series，将其转换为 DataFrame
            df = data.to_frame(name=data.name or 'Value').reset_index()
            # 如果 Series 的索引没有名字，默认为 'index'
            index_name = data.index.name or 'Index'
            df = df.rename(columns={'index': index_name})
        elif isinstance(data, pd.DataFrame):
            # 如果已经是 DataFrame，直接使用
            df = data
        else:
            # 如果是其他类型，报错
            self.error(f"table() 方法只接受 pandas DataFrame 或 Series 作为输入，但收到了 {type(data)}。")
            return
            
        rich_table = Table(title=f"[bold]{title}[/bold]", show_lines=True)
        
        # 添加列
        for col in df.columns:
            # 尝试将数值类型的列右对齐
            justify = "right" if pd.api.types.is_numeric_dtype(df[col]) else "left"
            style = "magenta" if justify == "right" else "cyan"
            rich_table.add_column(col, justify=justify, style=style, no_wrap=False)
            
        # 添加行
        for _, row in df.iterrows():
            row_str = []
            for item, col in zip(row.values, df.columns):
                # 如果是浮点数，格式化为6位小数
                if isinstance(item, float):
                    row_str.append(f"{item:.6f}")
                else:
                    row_str.append(str(item))
            rich_table.add_row(*row_str)
            
        self.console.print(rich_table)
        # 同时写入文件
        text = self._capture_rich_text(rich_table)
        self._write_to_file(text)

    def print(self, *objects, **kwargs):
        """
        直接调用底层的 Rich Console.print() 方法。
        用于打印不受日志级别控制的、需要精细格式化的内容。
        这不会有任何日志前缀 (如 INFO, WARNING)。
        """
        self.console.print(*objects, **kwargs)
        # 同时写入文件
        text = self._capture_rich_text(*objects, **kwargs)
        self._write_to_file(text)


    def progress(self, sequence: Iterable, description: str = "Processing..."):
        """返回一个可迭代的进度条对象。"""
        return track(sequence, description=description, console=self.console)
        
    def spacer(self, lines: int = 1):
        """打印一个或多个空行，用于增加垂直间距。"""
        for _ in range(lines):
            self.console.print()
            # 同时写入文件
            self._write_to_file("")

logger = RichLogger(name="mizar", log_file="pipeline.log")

# 使用示例：
# 1. 初始化时设置文件路径
# logger_custom = RichLogger(name="my_app", log_file="custom.log")

# 2. 动态修改文件路径
# logger.set_log_file("new_log_file.log")

# 3. 关闭文件日志
# logger.set_log_file(None)

# 4. 一次性配置多个参数
# logger.configure(verbose=True, log_file="combined.log", level=logging.DEBUG)