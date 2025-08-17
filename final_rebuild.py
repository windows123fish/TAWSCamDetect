import subprocess
import os
import sys

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"当前目录: {current_dir}")

# 检查Python解释器路径
python_path = sys.executable
print(f"Python解释器路径: {python_path}")

# 构建pyinstaller命令
spec_file = os.path.join(current_dir, 'main.spec')
command = [python_path, '-m', 'PyInstaller', spec_file]

# 运行命令
print(f"运行命令: {' '.join(command)}")
try:
    subprocess.run(command, check=True, cwd=current_dir)
    print("打包成功！")
except subprocess.CalledProcessError as e:
    print(f"打包失败，错误码: {e.returncode}")
except Exception as e:
    print(f"发生错误: {e}")

input("按Enter键退出...")