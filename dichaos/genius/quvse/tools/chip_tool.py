import asyncio, pdb
import pandas as pd
from dichaos.services.react.tool import BaseTool, ToolResult

try:
    import akshare as ak
except ImportError:
    ak = None


class ChipAnalysisTool(BaseTool):
    name: str = "chip_analysis_tool"
    description: str = "获取指定股票最新的筹码分布核心指标。"
    parameters: dict = {
        "type": "object",
        "properties": {
            "stock_code": {
                "type": "string",
                "description": "股票代码, 例如 '600519'。"
            }
        },
        "required": ["stock_code"],
    }

    async def execute(self, stock_code: str) -> ToolResult:
        if ak is None: return ToolResult(error="akshare 库未安装，无法执行工具。")
        print(f"⚙️  [Tool Server] 接收到请求: 分析股票 {stock_code} 的筹码...")
        try:
            # Run the synchronous akshare call in a separate thread
            df = await asyncio.to_thread(ak.stock_cyq_em, symbol=stock_code)

            if df is None or df.empty:
                return ToolResult(error=f"未能获取到 {stock_code} 的筹码数据。")

            # Data processing to find the latest data
            if '日期' in df.columns:
                df['日期'] = pd.to_datetime(df['日期'])
                df = df.sort_values(by='日期', ascending=True)
            else:
                return ToolResult(error="akshare 返回的数据缺少 '日期' 列。")

            latest_data = df.iloc[-1]
            date_str = latest_data['日期'].strftime('%Y-%m-%d')

            result = {
                "日期": date_str,
                "平均成本": f"{latest_data.get('平均成本', 0):.2f}",
                "获利比例": f"{latest_data.get('获利比例', 0):.2f}%",
                "90%成本集中度": f"{latest_data.get('90%成本集中度', 0):.2f}%",
            }
            print(f"✅ [Tool Server] 分析完成，返回结果: {result}")
            return ToolResult(output=result)
        except Exception as e:
            error_msg = f"在工具内部执行时发生错误: {e.__class__.__name__}: {e}"
            print(f"❌ [Tool Server] {error_msg}")
            return ToolResult(error=error_msg)
