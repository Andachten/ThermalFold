from pathlib import Path
import subprocess

PROJECT_NAME = "ThermalFold"
PYTHON_VERSION = ">=3.10"
SRC_PATH = Path("src") / PROJECT_NAME

def create_src_layout():
    """创建 src 目录和基础文件"""
    SRC_PATH.mkdir(parents=True, exist_ok=True)
    init_file = SRC_PATH / "__init__.py"
    if not init_file.exists():
        init_file.write_text(f"# {PROJECT_NAME} package init", encoding="utf-8")
    if not Path("README.md").exists():
        Path("README.md").write_text(f"# {PROJECT_NAME}\nProject description here.\n", encoding="utf-8")
    print("✅ src layout and base files ready.")

def init_poetry_project():
    """初始化 Poetry 项目，只生成 pyproject.toml，不添加依赖"""
    subprocess.run([
        "poetry", "init",
        "--no-interaction",
        "--name", PROJECT_NAME,
        "--description", f"{PROJECT_NAME} project",
        "--author", "Your Name <you@example.com>",
        "--python", PYTHON_VERSION
    ], check=True)
    print("✅ poetry init done (no dependencies added).")

def configure_src_layout_in_pyproject():
    """在 pyproject.toml 中添加 src layout 配置"""
    pyproj_path = Path("pyproject.toml")
    if not pyproj_path.exists():
        print("❌ pyproject.toml not found!")
        return

    content = pyproj_path.read_text(encoding="utf-8")
    if "[tool.setuptools.packages.find]" not in content:
        content += f"""

[tool.setuptools.packages.find]
where = ["src"]
include = ["{PROJECT_NAME}*"]
exclude = ["tests*"]
"""
        pyproj_path.write_text(content, encoding="utf-8")
        print("✅ src layout configured in pyproject.toml")

if __name__ == "__main__":
    create_src_layout()
    init_poetry_project()
    configure_src_layout_in_pyproject()