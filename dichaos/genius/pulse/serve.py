import sys, os, argparse, pdb
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from dichaos.services.mcp.server import create_app
from dichaos.services.mcp.registry import Registry
from tools.chip_tool import ChipAnalysisTool
from tools.factor import IdeaFactorTool
from tools.factor import ExpsFactorTool
from tools.factor import DSLFactorTool
from tools.factor import ForgeFactorTool

if __name__ == "__main__":
    registry = Registry()
    registry.add_tool(ChipAnalysisTool)
    registry.add_tool(IdeaFactorTool)
    registry.add_tool(ExpsFactorTool)
    registry.add_tool(DSLFactorTool)
    registry.add_tool(ForgeFactorTool)
    parser = argparse.ArgumentParser(description="Unified FastAPI Tool Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)  # 使用一个统一的端口，比如 8000
    args = parser.parse_args()
    all_tools = registry.all_tools()
    app = create_app(tool_instances=all_tools)

    # 3. 启动 uvicorn 服务器
    print(
        f"🚀 统一工具服务器已启动 ({len(all_tools)} 个工具已加载)，访问 http://{args.host}:{args.port}/docs 查看 API 文档"
    )
    uvicorn.run(app, host=args.host, port=args.port)
