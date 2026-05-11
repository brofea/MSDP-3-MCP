# 传统 SaaS 到 AI Agent (MCP) 全景实战

本项目实现了一个完整的实验，对比传统 SaaS 架构与 AI Agent 架构的差异。

## 📁 文件说明

| 文件名 | 说明 |
|--------|------|
| [app.py](app.py) | 主应用程序，包含完整的实现 |
| [test_app.py](test_app.py) | 单元测试和性能测试 |
| [requirements.txt](requirements.txt) | Python 依赖包列表 |
| [instruction.md](../instruction.md) | 详细的实验手册 |

## 🚀 快速开始

### 1. 创建 Conda 环境并安装依赖

```bash
conda create -y -n msd-agent-mcp python=3.10
conda activate msd-agent-mcp
python -m pip install -r requirements.txt
```

### 2. 启动 Ollama 服务

确保 Ollama 服务正在运行：

```bash
ollama serve
```

### 3. 下载模型

本实验使用 [unsloth 量化的 Qwen 3.6 27B GGUF 模型](https://www.modelscope.cn/models/unsloth/Qwen3.6-27B-GGUF/) 以及 [unsloth 量化的 Qwen 3.6 35B-A3B GGUF 模型](https://www.modelscope.cn/models/unsloth/Qwen3.6-35B-A3B-GGUF)

例如统一使用 `IQ2_M` 量化版本：

```bash
# 导航到 gguf 文件目录
ollama create qwen3.6:27b -f Qwen3.6-27B-UD-IQ2_M
ollama create qwen3.6:35b-a3b -f Qwen3.6-35B-A3B-UD-IQ2_M
ollama list
```

### 4. 运行应用

```bash
conda activate msd-agent-mcp
python app.py
```

应用启动后，访问 http://localhost:7860 即可使用。

## 🧪 运行测试

### 运行单元测试

```bash
conda activate msd-agent-mcp
python test_app.py
```

### 运行性能测试

```bash
conda activate msd-agent-mcp
python test_app.py --performance
```

### 验证 instruction.md 中的 Python 代码块

```bash
conda activate msd-agent-mcp
python verify_instruction_code.py

# 在 qwen3.6:35b-a3b 已下载后，可额外验证真实 Ollama API 连接片段
python verify_instruction_code.py --live-ollama
```

## 📊 功能特性

### 传统 SaaS 架构
- 硬编码的业务流程
- 高效但缺乏灵活性
- 一键执行完整工资计算流程

### AI Agent 架构
- 自然语言交互
- 自主规划工具调用
- 支持 Dense 和 MoE 两种模型
- 完整的日志记录

### 业务工具
- `get_employee_directory()` - 获取员工目录
- `calculate_payroll_and_tax()` - 计算工资和税款
- `export_payroll_csv()` - 导出 CSV 文件

## 🎯 使用示例

### SaaS 模式
点击"一键执行：生成工资单并下载"按钮，系统会自动：
1. 查询所有员工
2. 计算工资和税款
3. 导出 CSV 文件

### Agent 模式
在自然语言输入框中输入指令，例如：
- "帮我查一下全公司的工资，算好扣税，然后给我个 CSV 文件"
- "只看李四的工资"
- "计算张三的工资"

## 📝 日志查看

应用运行时会生成 `agent_orchestrator.log` 日志文件，包含：
- 用户请求记录
- 模型调用信息
- 工具执行时间
- 错误栈追踪

## 🐛 故障排除

### Ollama 服务未启动
```bash
# Windows：检查系统托盘中是否有 Ollama 图标
# macOS/Linux：运行 `ollama serve` 启动服务
```

### 模型未找到
```bash
# 检查模型是否已下载
ollama list

# 32GB 以上机器可以直接下载官方默认量化；16GB 机器请按 instruction.md 选择低内存 GGUF 后导入
ollama pull qwen3.6:35b-a3b
```

### 内存不足
- 关闭其他应用程序释放内存
- 16GB 机器主实验优先使用 MoE 35B-A3B 的 `IQ3/Q3` 低内存量化；Dense 27B 只作为短提示对照，必要时使用模型页面提供的 `IQ2/Q2_K` 兜底
- 32GB 以上机器再优先使用官方默认 `q4_K_M`，避免在普通课堂机器上下载 BF16 版本
- 低内存导入 GGUF 时建议设置 `PARAMETER num_ctx 2048`

## 📚 更多信息

详细的实验说明请查看 [instruction.md](../instruction.md)。

## 🎓 学习要点

通过本实验，同学们将学习：
- 传统 SaaS 架构与 AI Agent 架构的差异
- Dense 模型与 MoE 模型的对比
- MCP 工具定义规范
- Agent 调度器的实现
- 多工具链式调用

## 📄 许可证

本项目仅用于学术研究和教学目的。
