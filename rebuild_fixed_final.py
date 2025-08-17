import subprocess
import os

# 获取当前目录
current_dir = os.path.abspath('.')
print(f"当前工作目录: {current_dir}")

# 运行pyinstaller命令
try:
    # 使用shell=True来处理路径中的特殊字符
    subprocess.run("pyinstaller main.spec", check=True, cwd=current_dir, shell=True)
    print("打包成功！")
except subprocess.CalledProcessError as e:
    print(f"打包失败，错误码: {e.returncode}")
except Exception as e:
    print(f"发生错误: {e}")

input("按Enter键退出...")