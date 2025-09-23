import json, re, pdb
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from dichaos.kdutils import kd_logger  # 替换为您实际的日志记录器，或使用下面的
#from rich.console import Console
#kd_logger = Console()


def standardize_report_text(text: str) -> str:
    """
    统一报告文本中的标题格式和括号，使其标准化。
    """
    if not text:
        return ""

    text = re.sub(r'^\s*“?\*\*(.*?)\*\*”?\s*$',
                  r'#### \1',
                  text,
                  flags=re.MULTILINE)
    text = text.replace('[', '【').replace(']', '】')

    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith(
                '【步骤') and not stripped_line.startswith('####'):
            processed_lines.append(f"#### {stripped_line}")
        else:
            processed_lines.append(line)

    return '\n'.join(processed_lines)


class ReportGenerator(object):

    def __init__(self,
                 output_dir: str = "report_output_DiChaos",
                 template_name: str = "report_template.html"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.template_path = template_name#Path(__file__).parent / template_name

    def generate_and_save(self, report_data: Dict[str, Any]):
        symbol = report_data.get('symbol', 'UNKNOWN')
        kd_logger.info(
            f"🚀 [bold blue]DiChaos:[/bold blue] Generating report for [bold cyan]{symbol}[/bold cyan]..."
        )

        try:
            # 在这里调用模板渲染，而不是在 _get_html_template_with_data 中
            html_content = self._render_template(report_data)
        except FileNotFoundError:
            kd_logger.error(
                f"[bold red]❌ Template file not found at: {self.template_path}[/bold red]"
            )
            return  # 找不到模板则直接退出

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"DiChaos_Report_{symbol}.html"
        file_path = self.output_dir / filename

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            resolved_path = file_path.resolve()
            kd_logger.info(
                f"[bold green]✅ Report generated successfully![/bold green]")
            kd_logger.info(
                f"📄 Saved to: [link=file://{resolved_path}]{resolved_path}[/link]"
            )
        except Exception as e:
            kd_logger.print(f"[bold red]❌ Failed to save file: {e}[/bold red]")

    def _render_template(self, report_data: Dict[str, Any]) -> str:
        """
        读取HTML模板，并用数据替换占位符。
        """
        # 读取模板文件内容
        with open(self.template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()

        # 准备要注入的数据
        data_json_string = json.dumps(report_data,
                                      indent=2,
                                      ensure_ascii=False)
        current_year = str(datetime.now().year)

        # 替换占位符
        rendered_html = template_content.replace("{{ report_data_json }}",
                                                 data_json_string)
        rendered_html = rendered_html.replace("{{ current_year }}",
                                              current_year)

        return rendered_html

    def run(self, report_data):
        # 1. 清洗和标准化数据
        for agent, report in report_data.get('reason_reports', {}).items():
            report_data['reason_reports'][agent] = standardize_report_text(
                report)

        if 'reasoning' in report_data.get('final_prediction', {}):
            report_data['final_prediction'][
                'reasoning'] = standardize_report_text(
                    report_data['final_prediction']['reasoning'])

        # 2. 生成报告
        self.generate_and_save(report_data)
