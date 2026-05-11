"""课程文档第二章 instruction.md 中单元测试示例的落地版。

这些测试对应 instruction.md 的四类示例：
1. 工具函数测试；
2. Agent 调度器测试；
3. Gradio 启动集成测试；
4. 性能测试。

除 Gradio 启动外，所有测试都不真实调用 Ollama 大模型；Agent 测试使用 mock，避免
单元测试占用大量内存。
"""

import json
import os
import subprocess
import sys
import time
import unittest
from unittest.mock import Mock

import requests

import app
from app import (
    agent_orchestrator,
    calculate_payroll_and_tax,
    export_payroll_csv,
    get_employee_directory,
)


class TestInstructionTools(unittest.TestCase):
    """对应 instruction.md 的“工具函数测试”。"""

    def test_get_employee_directory(self):
        result = get_employee_directory()
        data = json.loads(result)

        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertIn("id", data[0])
        self.assertIn("name", data[0])
        self.assertIn("level", data[0])

    def test_calculate_payroll_and_tax_with_valid_data(self):
        employees = json.dumps([{"id": "E01", "name": "张三", "level": "L1"}], ensure_ascii=False)
        result = calculate_payroll_and_tax(employees)
        data = json.loads(result)

        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertIn("应发工资", data[0])
        self.assertIn("实发工资", data[0])
        self.assertEqual(data[0]["实发工资"], 7600)

    def test_calculate_payroll_and_tax_with_empty_data(self):
        result = calculate_payroll_and_tax("")
        data = json.loads(result)

        self.assertIn("error", data)

    def test_calculate_payroll_and_tax_with_invalid_json(self):
        result = calculate_payroll_and_tax("invalid json")
        data = json.loads(result)

        self.assertIn("error", data)

    def test_export_payroll_csv_with_valid_data(self):
        payroll_data = json.dumps([
            {"id": "E01", "name": "张三", "level": "L1", "应发工资": 10000, "实发工资": 8000}
        ], ensure_ascii=False)
        result = export_payroll_csv(payroll_data)
        data = json.loads(result)

        self.assertEqual(data.get("status"), "success")
        self.assertIn("file_path", data)
        self.assertTrue(os.path.exists(data["file_path"]))


class TestInstructionAgentOrchestrator(unittest.TestCase):
    """对应 instruction.md 的“Agent 调度器测试”。"""

    def setUp(self):
        self.original_client = app.client

    def tearDown(self):
        app.client = self.original_client

    def test_agent_orchestrator_with_simple_query(self):
        """非工资问题走普通 auto 分支；mock 模型直接返回最终文本。"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "这是测试回复"
        mock_response.choices[0].message.tool_calls = None

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        app.client = mock_client

        outputs = list(agent_orchestrator("测试问题", [], [], "qwen3.6:35b-a3b"))
        updated_history, updated_messages = outputs[-1]

        self.assertEqual(len(updated_history), 2)
        self.assertEqual(updated_history[-1]["role"], "assistant")
        self.assertIn("测试回复", updated_history[-1]["content"])
        self.assertGreaterEqual(len(updated_messages), 2)

    def test_agent_orchestrator_with_guided_tool_chain(self):
        """工资问题应通过三次 mock tool_call 完成工具链。"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            self._tool_response("call_1", "get_employee_directory", "{}"),
            self._tool_response("call_2", "calculate_payroll_and_tax", '{"employees_json":"PLACEHOLDER"}'),
            self._tool_response("call_3", "export_payroll_csv", '{"payroll_json":"PLACEHOLDER"}'),
        ]
        app.client = mock_client

        outputs = list(agent_orchestrator(
            "帮我查一下全公司的工资，算好扣税，然后给我个 CSV 文件",
            [],
            [],
            "qwen3.6:35b-a3b",
        ))
        updated_history, updated_messages = outputs[-1]
        final_text = updated_history[-1]["content"]

        self.assertIn("真实模型 tool call", final_text)
        self.assertIn("CSV 已导出", final_text)
        self.assertIn("张三", final_text)
        self.assertEqual(mock_client.chat.completions.create.call_count, 3)
        tool_messages = [message for message in updated_messages if message.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 3)

    @staticmethod
    def _tool_response(call_id, name, arguments):
        tool_call = Mock()
        tool_call.id = call_id
        tool_call.type = "function"
        tool_call.function = Mock()
        tool_call.function.name = name
        tool_call.function.arguments = arguments

        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        response.choices[0].message.content = None
        response.choices[0].message.tool_calls = [tool_call]
        return response


class TestInstructionIntegration(unittest.TestCase):
    """对应 instruction.md 的“集成测试”。"""

    def test_gradio_launch(self):
        """启动完整 Gradio 应用，并检查本项目默认端口返回 HTTP 200。"""
        port = "7961"
        env = os.environ.copy()
        env["GRADIO_SERVER_PORT"] = port
        process = subprocess.Popen(
            [sys.executable, "app.py"],
            cwd=os.path.dirname(__file__),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            response = self._wait_for_http_ok(f"http://127.0.0.1:{port}", timeout=25)
            self.assertEqual(response.status_code, 200)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    @staticmethod
    def _wait_for_http_ok(url, timeout):
        deadline = time.time() + timeout
        last_error = None
        while time.time() < deadline:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    return response
            except requests.RequestException as exc:
                last_error = exc
            time.sleep(0.5)
        raise AssertionError(f"Gradio 未在 {timeout} 秒内启动: {last_error}")


class TestInstructionPerformance(unittest.TestCase):
    """对应 instruction.md 的“性能测试”。"""

    def test_tool_functions_are_fast_enough(self):
        start_time = time.time()
        for _ in range(100):
            get_employee_directory()
        directory_avg = (time.time() - start_time) / 100

        employees = get_employee_directory()
        start_time = time.time()
        for _ in range(100):
            calculate_payroll_and_tax(employees)
        payroll_avg = (time.time() - start_time) / 100

        # 阈值故意宽松，只用于防止明显卡死，不做严肃性能基准。
        self.assertLess(directory_avg, 0.05)
        self.assertLess(payroll_avg, 0.05)


if __name__ == "__main__":
    unittest.main()
