import asyncio
from dichaos.services.react.tool import HTTPToolCollection
from typing import Any, Dict

class ToolExecutor:
    """
    一个轻量级的客户端，用于直接连接到工具服务器并执行指定的工具，
    完全绕过 Agent 和 ReAct 循环。
    """
    def __init__(self, server_url: str):
        """
        初始化执行器。
        :param server_url: 统一工具服务器的 URL, e.g., "http://127.0.0.1:8000"
        """
        self.server_url = server_url
        self.tool_collection = HTTPToolCollection()
        self._is_connected = False

    async def connect(self):
        """连接到工具服务器并发现可用工具。"""
        if not self._is_connected:
            print(f"🔗 [ToolExecutor] 正在连接到工具服务器 at {self.server_url}...")
            await self.tool_collection.connect(self.server_url)
            self._is_connected = True
            print("✅ [ToolExecutor] 连接成功。")

    async def disconnect(self):
        """断开与工具服务器的连接。"""
        if self._is_connected:
            await self.tool_collection.disconnect()
            self._is_connected = False
            print("🧹 [ToolExecutor] 已断开连接。")

    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        直接执行一个远程工具。
        
        :param tool_name: 要执行的工具的名称。
        :param kwargs: 传递给工具的参数。
        :return: 工具返回的 output 字段，如果出错则抛出异常。
        """
        if not self._is_connected:
            raise ConnectionError("Executor not connected. Please call .connect() first.")
        
        print(f"⚙️  [ToolExecutor] 正在直接调用工具 '{tool_name}'...")
        result = await self.tool_collection.execute(name=tool_name, tool_input=kwargs)
        
        if result.error:
            print(f"❌ [ToolExecutor] 工具 '{tool_name}' 执行失败: {result.error}")
            raise Exception(f"Tool execution failed: {result.error}")
            
        print(f"✅ [ToolExecutor] 工具 '{tool_name}' 执行成功。")
        return result.output

    # --- (可选) 异步上下文管理器支持 ---
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()