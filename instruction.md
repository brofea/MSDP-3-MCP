# 传统 SaaS 到 AI Agent (MCP) 全景实战傻瓜版 Instruction

> 本文档只面向本次实验实操。模型固定使用：
>
> - `qwen3.6:27b`：Dense 对照模型
> - `qwen3.6:35b-a3b`：MoE 主实验模型
>
> 目标不是重新造一个复杂系统，而是跑通一个最小闭环：传统 SaaS 固定流程 vs AI Agent 自主调用工具。

## 0. 你最终要做出什么

完成后，`2-MCP` 项目目录建议长这样：

```text
2-MCP/
├── Instruction.md
├── 报告参考.md
├── README.md
├── requirements.txt
├── app.py
├── test_app.py
└── verify_instruction_code.py
```

核心运行结果：

1. `python test_app.py` 能通过单元测试。
2. `python app.py` 能启动 Gradio 页面。
3. 页面左侧 SaaS 按钮能一键生成工资表。
4. 页面右侧 Agent 能分别选择 `qwen3.6:27b` 和 `qwen3.6:35b-a3b` 完成自然语言工资计算任务。
5. 报告中能截图证明 Ollama、模型、测试、UI 和 Agent 调用都成功。

## 1. 实验思路先讲人话

本实验只有一个业务场景：给员工算工资，并导出 CSV。

传统 SaaS 的做法是程序员写死流程：

```text
查员工 -> 算工资和税 -> 导出 CSV
```

AI Agent 的做法是把每一步拆成工具，让模型根据自然语言自己决定调用顺序：

```text
用户说：“帮我算全公司工资并导出”
模型判断：需要先查员工
程序执行：get_employee_directory()
模型看到员工列表后判断：需要算工资
程序执行：calculate_payroll_and_tax(...)
模型看到工资结果后判断：需要导出 CSV
程序执行：export_payroll_csv(...)
模型总结结果给用户
```

你要观察的不是“模型会不会聊天”，而是它能不能稳定完成：

```text
思考 -> 行动(tool call) -> 观察(tool result) -> 下一步行动 -> 最终回答
```

## 2. 准备 Ollama 和两个模型

### 2.1 检查 Ollama

```bash
ollama --version
ollama ps
```

作用：

- `ollama --version`：确认 Ollama 已安装。
- `ollama ps`：确认 Ollama 服务能响应。如果没有模型在运行，列表为空也没关系。

报告截图建议：截 `ollama --version` 输出。

### 2.2 确认两个模型名称

本实验代码默认只认下面两个名字：

```text
qwen3.6:27b
qwen3.6:35b-a3b
```

检查命令：

```bash
ollama list
```

如果列表里已经有这两个名字，就继续下一步。

如果没有，可以任选一种方式准备模型。

### 2.3 方式 A：直接从 Ollama 拉取

适合内存/统一内存 32GB 以上的电脑。

```bash
ollama pull qwen3.6:27b
ollama pull qwen3.6:35b-a3b
ollama list
```

作用：

- `ollama pull qwen3.6:27b`：下载 Dense 模型。
- `ollama pull qwen3.6:35b-a3b`：下载 MoE 模型。
- `ollama list`：确认模型已经在本机注册。

### 2.4 方式 B：从 ModelScope 或 GGUF 导入后改成实验名称

如果你下载到的是 ModelScope 长模型名，最后一定要复制成短名字：

```bash
ollama cp "你的ModelScope-27B模型名" qwen3.6:27b
ollama cp "你的ModelScope-35B-A3B模型名" qwen3.6:35b-a3b
ollama list
```

作用：

- `ollama cp` 不是复制大文件，而是在 Ollama 中给模型增加一个新标签。
- 这样 `app.py` 里的模型下拉框不用改。

如果你只有 `.gguf` 文件，就创建 `Modelfile`。

`Modelfile-35b-a3b` 示例：

```text
FROM /你的路径/Qwen3.6-35B-A3B-IQ3.gguf
PARAMETER num_ctx 2048
PARAMETER num_predict 512
```

然后导入：

```bash
ollama create qwen3.6:35b-a3b -f Modelfile-35b-a3b
```

`Modelfile-27b` 示例：

```text
FROM /你的路径/Qwen3.6-27B-IQ3.gguf
PARAMETER num_ctx 2048
PARAMETER num_predict 512
```

然后导入：

```bash
ollama create qwen3.6:27b -f Modelfile-27b
```

作用：

- `FROM`：告诉 Ollama 使用哪个本地 GGUF 权重文件。
- `PARAMETER num_ctx 2048`：限制上下文长度，降低内存压力。
- `PARAMETER num_predict 512`：限制一次最多生成 512 个 token，避免验证阶段跑太久。
- `ollama create`：把 GGUF 文件注册成 Ollama 可以运行的模型。

### 2.5 单独测试两个模型

```bash
ollama run qwen3.6:27b "你好，请用一句话说明你已成功加载。"
ollama run qwen3.6:35b-a3b "你好，请用一句话说明你已成功加载。"
```

作用：

- 确认模型不仅在列表里，而且真的能推理。
- 两个命令都建议截图，报告里分别作为 Dense 和 MoE 模型测试证据。

## 3. 创建 Python 环境

建议使用 Conda，避免污染系统 Python。

```bash
conda create -y -n msd-agent-mcp python=3.10
conda activate msd-agent-mcp
python --version
python -m pip --version
```

作用：

- `conda create -y -n msd-agent-mcp python=3.10`：创建名为 `msd-agent-mcp` 的课程环境。
- `conda activate msd-agent-mcp`：进入这个环境。
- `python --version`：确认 Python 是 3.10。
- `python -m pip --version`：确认 pip 指向当前 Conda 环境。

## 4. 准备项目文件

你可以直接把课程附件复制到 `/2-MCP`，再基于这些文件完成实验。

在仓库根目录执行：

```bash
cp "课程文档/2. 传统 SaaS 到 AI Agent (MCP) 全景实战/附件/app.py" "2-MCP/app.py"
cp "课程文档/2. 传统 SaaS 到 AI Agent (MCP) 全景实战/附件/test_app.py" "2-MCP/test_app.py"
cp "课程文档/2. 传统 SaaS 到 AI Agent (MCP) 全景实战/附件/requirements.txt" "2-MCP/requirements.txt"
cp "课程文档/2. 传统 SaaS 到 AI Agent (MCP) 全景实战/附件/README.md" "2-MCP/README.md"
cp "课程文档/2. 传统 SaaS 到 AI Agent (MCP) 全景实战/附件/verify_instruction_code.py" "2-MCP/verify_instruction_code.py"
```

作用：

- 把老师提供的可运行代码放到本次提交目录 `2-MCP`。
- 后续截图、测试、报告都围绕 `2-MCP` 做，不再改 `课程文档`。

进入项目目录：

```bash
cd "2-MCP"
```

安装依赖：

```bash
conda activate msd-agent-mcp
python -m pip install -r requirements.txt
```

`requirements.txt` 中主要依赖：

- `gradio`：做本地网页 UI。
- `openai`：用 OpenAI 兼容 SDK 调用 Ollama 的本地接口。
- `requests`：用于接口连通性或扩展测试。

## 5. app.py 代码逐块解释

### 5.1 导入库

```python
import json
import os
import time
import csv
import tempfile
import logging
from datetime import datetime
```

作用：

- `json`：工具输入输出统一使用 JSON 字符串。
- `os`：处理文件路径。
- `time`：统计工具执行耗时，模拟 SaaS 处理时间。
- `csv`：导出工资 CSV 文件。
- `tempfile`：把 CSV 写到系统临时目录，避免手动配置输出目录。
- `logging`：记录 Agent 每一步调用了什么工具、耗时多少。
- `datetime`：保留给时间相关扩展使用。

```python
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
```

作用：

- 尝试导入 OpenAI SDK。
- 如果没有安装 `openai`，工具函数和测试仍可以运行，模型调用部分不可用。

### 5.2 日志配置

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_orchestrator.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
```

作用：

- 日志写两份：一份到 `agent_orchestrator.log`，一份打印到终端。
- 你可以在报告中截图日志，证明 Agent 确实调用了工具。

### 5.3 初始化 Ollama 客户端

```python
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='local',
    timeout=120
)
```

作用：

- Ollama 默认运行在 `http://localhost:11434`。
- `/v1` 表示使用 OpenAI 兼容接口。
- `api_key='local'` 对本地 Ollama 没有鉴权意义，只是 SDK 要求必须填。
- `timeout=120` 表示最多等模型 120 秒，避免大模型首 token 慢时过早报错。

### 5.4 模拟数据

```python
mock_employees = [
    {"id": "E01", "name": "张三", "level": "L1"},
    {"id": "E02", "name": "李四", "level": "L2"},
    {"id": "E03", "name": "王五", "level": "L3"}
]

mock_salary_levels = {"L1": 10000, "L2": 20000, "L3": 35000}
```

作用：

- `mock_employees` 模拟员工表。
- `mock_salary_levels` 模拟职级到工资的映射。
- 本实验不连接真实数据库，避免数据安全和环境复杂度问题。

### 5.5 原子工具 1：查询员工目录

```python
def get_employee_directory():
    return json.dumps(mock_employees, ensure_ascii=False)
```

作用：

- 返回所有员工的基础信息。
- 无需参数，适合作为工具链第一步。
- `ensure_ascii=False` 用来保留中文，不把中文转义成 `\u5f20\u4e09`。

实际附件中还做了异常处理：如果员工数据为空，会返回 `{"error": "员工数据为空"}`。

### 5.6 原子工具 2：计算工资和税

```python
def calculate_payroll_and_tax(employees_json: str):
    employees = json.loads(employees_json)
    ...
```

作用：

- 输入员工 JSON 字符串。
- 解析后根据 `level` 找到基础工资。
- 按简化规则计算：
  - 五险一金 = 基础工资 x 20%
  - 个税 = (基础工资 - 五险一金) x 5%
  - 实发工资 = 基础工资 - 五险一金 - 个税

以张三 `L1` 为例：

```text
应发工资 = 10000
五险一金 = 10000 * 20% = 2000
个税 = (10000 - 2000) * 5% = 400
实发工资 = 10000 - 2000 - 400 = 7600
```

为什么这个函数输入输出都用 JSON 字符串？

- 大模型工具调用传参本质上是文本/JSON。
- 用 JSON 字符串可以让工具结果直接回传给模型，再作为下一步工具参数。

### 5.7 原子工具 3：导出 CSV

```python
def export_payroll_csv(payroll_json: str):
    payroll_data = json.loads(payroll_json)
    filepath = os.path.join(tempfile.gettempdir(), "payroll_report.csv")
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=payroll_data[0].keys())
        writer.writeheader()
        writer.writerows(payroll_data)
```

作用：

- 输入工资计算结果 JSON。
- 写成 CSV 文件。
- 返回文件路径、状态和记录数。

为什么写到临时目录？

- Windows、macOS、Linux 都能找到临时目录。
- 不需要你手动创建输出文件夹。

### 5.8 SaaS 控制器

```python
def saas_generate_payroll_api():
    emp_str = get_employee_directory()
    payroll_str = calculate_payroll_and_tax(emp_str)
    export_result = json.loads(export_payroll_csv(payroll_str))
    ...
```

作用：

- 传统 SaaS 的固定流水线。
- 不理解自然语言。
- 不会自己调整流程。
- 用户只能点击按钮，程序按写死顺序执行。

这部分对应报告里的“SaaS 对照组实现”。

### 5.9 MCP 风格 tools_schema

```python
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_employee_directory",
            "description": "第一步：获取全公司所有员工的基础数据（包含姓名和职级）。不需要参数。"
        }
    },
    ...
]
```

作用：

- 告诉模型：系统里有哪些工具。
- 模型看不到 Python 函数内部，只能看到工具名、描述和参数结构。
- 这就是本实验所谓的 MCP 风格工具定义。

一个工具 Schema 重点字段：

- `name`：工具名称，后端靠它路由到真实函数。
- `description`：告诉模型什么时候用这个工具。
- `parameters`：告诉模型参数是什么类型。
- `required`：告诉模型哪些参数必须提供。

注意：

- 本实验没有搭建完整 MCP Server。
- 本实验实现的是 MCP 的核心思想：用标准化 Schema 描述工具，让模型发起结构化调用，再由程序执行。

### 5.10 Agent 调度器

```python
def agent_orchestrator(user_message, history, messages_state, selected_model):
    ...
```

参数作用：

- `user_message`：用户输入的自然语言。
- `history`：Gradio 页面上显示的聊天历史。
- `messages_state`：真正发给模型的上下文，包含 system、user、assistant、tool 消息。
- `selected_model`：当前选择的模型，即 `qwen3.6:27b` 或 `qwen3.6:35b-a3b`。

核心流程：

```text
1. 把用户输入加入 messages_state
2. 调用模型，让模型决定是否需要工具
3. 如果模型返回 tool_calls：
   3.1 解析工具名和参数
   3.2 调用本地 Python 函数
   3.3 把工具结果以 role=tool 放回上下文
   3.4 继续下一轮
4. 如果模型没有返回 tool_calls：
   说明模型认为任务完成，输出最终回答
5. 最多循环 10 轮，防止死循环
```

关键代码：

```python
response = client.chat.completions.create(
    model=selected_model,
    messages=messages_state,
    tools=tools_schema,
    tool_choice="auto"
)
```

作用：

- `model=selected_model`：使用页面下拉框选中的模型。
- `messages=messages_state`：把完整上下文发给模型。
- `tools=tools_schema`：把工具菜单发给模型。
- `tool_choice="auto"`：让模型自己决定是否调用工具。

本地路由代码：

```python
if func_name == "get_employee_directory":
    tool_result = get_employee_directory()
elif func_name == "calculate_payroll_and_tax":
    tool_result = calculate_payroll_and_tax(employees_json=func_args.get("employees_json", "[]"))
elif func_name == "export_payroll_csv":
    tool_result = export_payroll_csv(payroll_json=func_args.get("payroll_json", "[]"))
```

作用：

- 模型只提出“我要调用哪个工具，参数是什么”。
- 真正执行工具的是 Python 程序。
- 这也是 Agent 系统的安全边界：模型不能直接运行任意代码。

### 5.11 Gradio UI

```python
def create_demo():
    import gradio as gr
    with gr.Blocks() as demo:
        ...
```

作用：

- 构建本地网页界面。
- 左侧是 SaaS 对照组。
- 右侧是 Agent 实验组。

核心组件：

- `gr.Button`：SaaS 一键执行按钮。
- `gr.Dataframe`：显示工资表。
- `gr.File`：提供导出的 CSV 文件。
- `gr.Dropdown`：选择 Dense 或 MoE 模型。
- `gr.Chatbot`：显示 Agent 对话和工具调用过程。
- `gr.Textbox`：输入自然语言任务。

模型选择器：

```python
model_selector = gr.Dropdown(
    choices=["qwen3.6:27b", "qwen3.6:35b-a3b"],
    value="qwen3.6:35b-a3b",
    label="核心变量：选择底层大模型架构 (Dense vs MoE)"
)
```

作用：

- 保证 Dense 和 MoE 对比使用同一套业务代码、同一套 UI、同一套工具 Schema。
- 唯一变量是底层模型。

### 5.12 启动应用

```python
if __name__ == "__main__":
    demo.launch(theme="soft")
```

作用：

- 当你运行 `python app.py` 时启动 Gradio。
- 终端会显示类似 `http://127.0.0.1:7860` 的地址。

## 6. test_app.py 测试说明

运行：

```bash
python test_app.py
```

作用：

- 验证三个工具函数是否正常。
- 验证 SaaS 控制器是否能跑完整流程。
- 验证端到端链路：查员工 -> 算工资 -> 导出 CSV。

你会看到类似：

```text
Ran 11 tests in ...
OK
```

报告截图建议：截 `python test_app.py` 的 `OK` 输出。

性能测试：

```bash
python test_app.py --performance
```

作用：

- 粗略观察纯 Python 工具函数的耗时。
- 说明本实验主要耗时来自模型推理，而不是业务函数。

## 7. 启动完整应用

确保 Ollama 正在运行，模型已经存在：

```bash
ollama list
```

进入项目目录：

```bash
cd "2-MCP"
conda activate msd-agent-mcp
python app.py
```

打开终端显示的网址，一般是：

```text
http://127.0.0.1:7860
```

## 8. 页面上怎么操作

### 8.1 测 SaaS 对照组

点击左侧：

```text
一键执行：生成工资单并下载
```

期望结果：

- 表格出现张三、李四、王五的工资。
- 文件区域出现 `payroll_report.csv`。

截图建议：

- 截左侧表格和文件输出。

### 8.2 测 Agent + MoE

右侧模型选择：

```text
qwen3.6:35b-a3b
```

输入：

```text
帮我查一下全公司的工资，算好扣税，然后给我个 CSV 文件。
```

期望结果：

- Agent 依次触发 `get_employee_directory`、`calculate_payroll_and_tax`、`export_payroll_csv`。
- 最终回答里有工资表或文件路径。

截图建议：

- 截模型选择框、输入内容、Agent 输出。
- 终端或 `agent_orchestrator.log` 中截工具调用日志。

### 8.3 测 Agent + Dense

右侧模型选择：

```text
qwen3.6:27b
```

输入同一句：

```text
帮我查一下全公司的工资，算好扣税，然后给我个 CSV 文件。
```

观察：

- 是否能按正确顺序调用工具。
- 是否能把上一步 JSON 正确传给下一步。
- 是否出现参数缺失、JSON 格式错误、循环调用、最终回答不完整等问题。

截图建议：

- 截 Dense 模型测试结果。
- 如果 Dense 慢或失败，也可以如实记录，这不一定扣分，重点是说明现象和原因。

## 9. 推荐 Git 提交流程

评分标准要求至少 5 次有意义 commit。可以这样做：

```bash
git init
git add README.md requirements.txt
git commit -m "chore: initialize mcp experiment project"

git add app.py
git commit -m "feat: add payroll tools and saas controller"

git add app.py
git commit -m "feat: add mcp-style tool schema and agent orchestrator"

git add test_app.py
git commit -m "test: add payroll workflow tests"

git add Instruction.md 报告参考.md
git commit -m "docs: add simplified instruction and report reference"
```

作用：

- `git init`：把 `2-MCP` 初始化成 Git 仓库。
- `git add`：把文件加入暂存区。
- `git commit`：保存一次有意义的版本。

如果你的大仓库已经是 Git 仓库，不需要重复 `git init`，直接在根目录提交也可以。

## 10. 报告截图清单

建议至少准备这些截图：

1. `ollama --version`
2. `ollama list` 中出现 `qwen3.6:27b` 和 `qwen3.6:35b-a3b`
3. `ollama run qwen3.6:27b ...` 成功回复
4. `ollama run qwen3.6:35b-a3b ...` 成功回复
5. `2-MCP` 项目目录结构
6. `python test_app.py` 测试通过
7. SaaS 左侧按钮生成工资表
8. Agent 使用 `qwen3.6:35b-a3b` 完成任务
9. Agent 使用 `qwen3.6:27b` 完成任务
10. `agent_orchestrator.log` 或终端中显示工具调用链
11. `git log --oneline` 至少 5 次提交

## 11. 常见问题

### 11.1 `model not found`

原因：Ollama 里没有对应模型名。

处理：

```bash
ollama list
```

确认名称必须完全匹配：

```text
qwen3.6:27b
qwen3.6:35b-a3b
```

如果是长名称，用：

```bash
ollama cp "长模型名" qwen3.6:27b
ollama cp "长模型名" qwen3.6:35b-a3b
```

### 11.2 `Connection refused`

原因：Ollama 服务没启动。

处理：

```bash
ollama ps
```

macOS/Linux 可尝试：

```bash
ollama serve
```

Windows 通常从开始菜单或任务栏启动 Ollama。

### 11.3 模型很慢

可能原因：

- 机器内存不足。
- 模型量化太高。
- 首次加载模型需要时间。
- 同时开了太多应用。

处理建议：

- 一次只运行一个模型。
- 16GB 机器优先使用低内存 GGUF，比如 IQ3/Q3。
- 关闭大浏览器标签、游戏、视频、虚拟机。
- 报告中如实记录慢的现象和硬件原因。

### 11.4 Agent 不调用工具，只聊天

处理：

- 换更明确的提示词：

```text
请必须使用工具完成：先调用 get_employee_directory 获取员工，再调用 calculate_payroll_and_tax 计算工资，最后调用 export_payroll_csv 导出 CSV。
```

- 优先用 `qwen3.6:35b-a3b` 测主线。
- 查看 `agent_orchestrator.log`，确认模型有没有返回 `tool_calls`。

### 11.5 JSON 参数错误

现象：

- Agent 调用第二个工具失败。
- 日志里出现 JSON 解析失败。

解释：

- 模型把上一步工具输出改坏了，或没有按 Schema 传字符串参数。
- 这是工具调用实验中很重要的观察点，可以写进 Dense/MoE 对比分析。

## 12. 最小验收流程

如果时间很紧，按这个顺序做：

```bash
ollama --version
ollama list
ollama run qwen3.6:35b-a3b "你好"
ollama run qwen3.6:27b "你好"

cd "2-MCP"
conda activate msd-agent-mcp
python -m pip install -r requirements.txt
python test_app.py
python app.py
```

然后在网页里：

1. 点左侧 SaaS 按钮。
2. 右侧选择 `qwen3.6:35b-a3b`，输入完整工资任务。
3. 右侧选择 `qwen3.6:27b`，输入同样任务。
4. 截图并整理到报告。
