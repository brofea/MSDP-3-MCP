"""项目自检脚本。

本脚本用于实验提交前做一次本地自动检查：
1. 必需文件是否齐全；
2. app.py 是否能编译和导入；
3. 工具函数、SaaS 控制器、端到端流程是否通过单元测试；
4. 可选检查 Ollama 中是否存在 qwen3.6:27b 与 qwen3.6:35b-a3b。

运行：
    python verify_instruction_code.py
    python verify_instruction_code.py --live-ollama
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "app.py",
    "test_app.py",
    "instruction.md",
    "instruction.html",
]
REQUIRED_MODELS = ["qwen3.6:27b", "qwen3.6:35b-a3b"]


def check_required_files() -> None:
    """确认评分清单要求的项目文件都在当前目录。"""
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).exists()]
    if missing:
        raise AssertionError(f"缺少必需文件: {', '.join(missing)}")
    print("OK: 必需文件齐全")


def check_app_compile() -> None:
    """使用 Python 编译器检查 app.py 语法。"""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(ROOT / "app.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise AssertionError("app.py 编译失败")
    print("OK: app.py 语法检查通过")


def check_business_smoke() -> None:
    """不依赖 Gradio/Ollama，直接验证核心业务闭环。"""
    sys.path.insert(0, str(ROOT))
    app = importlib.import_module("app")

    employees_json = app.get_employee_directory()
    employees = json.loads(employees_json)
    assert isinstance(employees, list) and len(employees) == 3

    payroll_json = app.calculate_payroll_and_tax(employees_json)
    payroll = json.loads(payroll_json)
    assert isinstance(payroll, list) and len(payroll) == 3
    assert payroll[0]["实发工资"] == 7600

    exported = json.loads(app.export_payroll_csv(payroll_json))
    assert exported["status"] == "success"
    assert exported["record_count"] == 3
    assert os.path.exists(exported["file_path"])

    table, path = app.saas_generate_payroll_api()
    assert len(table) == 3
    assert path and os.path.exists(path)

    assert getattr(app, "demo", None) is not None
    print("OK: 业务烟测通过")


def check_unit_tests() -> None:
    """运行 test_app.py 中的 unittest 测试集。"""
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT), pattern="test_app.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError("单元测试失败")
    print("OK: unittest 全部通过")


def check_ollama_models() -> None:
    """可选检查：确认两个实验模型已经注册到 Ollama。"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AssertionError(f"无法执行 ollama list: {exc}") from exc

    if result.returncode != 0:
        raise AssertionError(result.stderr.strip() or "ollama list 执行失败")

    missing = [model for model in REQUIRED_MODELS if model not in result.stdout]
    if missing:
        raise AssertionError(f"Ollama 中缺少模型: {', '.join(missing)}")
    print("OK: Ollama 模型检查通过")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-ollama", action="store_true", help="检查 Ollama 中是否存在两个实验模型")
    args = parser.parse_args()

    check_required_files()
    check_app_compile()
    check_business_smoke()
    check_unit_tests()
    if args.live_ollama:
        check_ollama_models()

    mode = "含 Ollama 模型检查" if args.live_ollama else "未检查 Ollama 模型"
    print(f"全部自检通过（{mode}）")


if __name__ == "__main__":
    main()
