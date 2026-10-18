import re, copy


class ExpressionNode(object):
    """
    表示表达式树中的一个节点。
    """

    def __init__(self, value, node_type):
        self.value = value  # 节点的值 (e.g., 'MIChimoku', 2, 'dv001...')
        self.node_type = node_type  # 'function', 'parameter', 'feature'
        self.children = []  # 子节点列表

    def add_child(self, node):
        self.children.append(node)

    def __str__(self):
        """将节点和其子树转换回表达式字符串"""
        if self.node_type in ['parameter', 'feature']:
            # 对特征字符串加上引号
            return f"'{self.value}'" if self.node_type == 'feature' else str(
                self.value)

        # 如果是函数节点
        args_str = ", ".join(str(child) for child in self.children)
        return f"{self.value}({args_str})"

    def extract_components(self):
        """
        递归地从节点及其子树中提取所有组件。
        """
        components = {'functions': [], 'parameters': [], 'features': []}

        if self.node_type == 'function':
            components['functions'].append(self.value)
        elif self.node_type == 'parameter':
            components['parameters'].append(self.value)
        elif self.node_type == 'feature':
            components['features'].append(self.value)

        for child in self.children:
            child_components = child.extract_components()
            components['functions'].extend(child_components['functions'])
            components['parameters'].extend(child_components['parameters'])
            components['features'].extend(child_components['features'])

        return components


class ExpressionParser(object):
    """
    一个简单的递归下降解析器，用于处理嵌套的因子表达式。
    """

    def __init__(self, expression_string):
        self.expression = expression_string.strip()
        # 使用正则表达式来匹配函数调用模式: function_name(arguments)
        # re.match 只从字符串开头匹配
        self.match = re.match(r"^\s*([a-zA-Z0-9_]+)\s*\((.*)\)\s*$",
                              self.expression)

    def _parse_arguments(self, args_string):
        """
        一个辅助函数，用于解析括号内的参数列表。
        它能正确处理嵌套的函数调用。
        """
        args = []
        current_arg = ""
        paren_level = 0
        in_string = False

        for char in args_string:
            if char == ',' and paren_level == 0 and not in_string:
                args.append(current_arg.strip())
                current_arg = ""
            else:
                if char == '(':
                    paren_level += 1
                elif char == ')':
                    paren_level -= 1
                elif char == "'":
                    in_string = not in_string
                current_arg += char

        if current_arg:
            args.append(current_arg.strip())

        return args

    def parse(self):
        """
        主解析方法，递归地构建表达式树。
        """
        if not self.match:
            # 如果不匹配函数模式，那么它就是一个参数或特征
            val = self.expression
            if val.startswith("'") and val.endswith("'"):
                # 如果有引号，是特征
                return ExpressionNode(val[1:-1], 'feature')
            else:
                # 否则是数字参数
                # 尝试转换为数字类型
                try:
                    return ExpressionNode(int(val), 'parameter')
                except ValueError:
                    try:
                        return ExpressionNode(float(val), 'parameter')
                    except ValueError:
                        # 如果转换失败，可能是个变量名，我们暂时也视为特征
                        return ExpressionNode(val, 'feature')

        # 如果匹配函数模式
        func_name = self.match.group(1)
        args_str = self.match.group(2)

        root = ExpressionNode(func_name, 'function')

        # 递归地解析每个参数
        arguments = self._parse_arguments(args_str)
        for arg in arguments:
            # 对每个参数字符串，创建一个新的解析器并解析
            child_parser = ExpressionParser(arg)
            child_node = child_parser.parse()
            root.add_child(child_node)

        return root


class FormulaConverter:
    """
    一个能够将因子公式字符串，动态地、不依赖预生成实例地，
    转换为您的Program框架所需的“前缀表示法”列表的转换器。
    """

    def __init__(self, operator_template_map):
        """
        初始化转换器。

        :param operator_template_map: dict, 一个从函数名字符串到其“模板”Function对象的映射。
        """
        self.template_map = operator_template_map

    def _find_and_instantiate_operator(self, func_name, param_nodes):
        """
        根据函数名和参数节点，查找模板并动态创建一个带正确参数的Function实例。
        """
        template = self.template_map.get(func_name)
        if template is None:
            raise ValueError(f"在模板库中找不到函数: '{func_name}'")

        # 这是一个关键的假设：我们根据您的Function结构，认为其核心参数
        # (如周期) 是第一个parameter类型的子节点，并将其值赋给新实例的default_value。
        # 如果您的函数有多个参数，这里的逻辑需要扩展。
        new_default_value = None
        if param_nodes:
            new_default_value = param_nodes[0].value

        # 使用模板克隆并创建带有正确参数的新实例
        factors_instance = copy.deepcopy(
            template)  #.clone_with_params(new_default_value=new_default_value)
        factors_instance.default_value = new_default_value
        return factors_instance

    def _tree_to_prefix_recursive(self, node: ExpressionNode):
        """
        【核心】递归地将AST树转换为您的Program期望的前缀表示法列表。
        """
        # --- 基本情况: 如果是叶子节点 (特征或参数)，直接返回值 ---
        if not node.children:
            return [node.value]

        # --- 递归情况: 如果是函数节点 ---
        prefix_list = []

        # 1. 首先，添加当前的操作符 (动态实例化的Function对象)
        if node.node_type == 'function':
            param_nodes = [
                child for child in node.children
                if child.node_type == 'parameter'
            ]
            function_instance = self._find_and_instantiate_operator(
                node.value, param_nodes)
            prefix_list.append(function_instance)
        else:
            # 理论上不会发生，因为有子节点的都是函数节点
            raise TypeError(f"节点 {node.value} 有子节点但不是函数类型。")

        # 2. 然后，严格按照原始顺序，递归地添加所有子节点的列表
        for child in node.children:
            prefix_list.extend(self._tree_to_prefix_recursive(child))

        return prefix_list

    def formula_to_program_list(self, formula_string: str):
        """
        主接口函数：将完整的因子公式字符串转换为可被您Program框架执行的列表。
        """
        if not isinstance(formula_string, str) or not formula_string:
            raise ValueError("输入的公式必须是一个非空字符串。")

        # 步骤A：先用通用解析器将字符串解析成AST树
        parser = ExpressionParser(formula_string)
        ast_tree = parser.parse()

        # 步骤B：再用我们的递归转换器将AST树转换为Program所需的“前缀表示法”列表
        program_list = self._tree_to_prefix_recursive(ast_tree)

        return program_list
