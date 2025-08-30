import os
import subprocess

# 切换到当前目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 执行PyInstaller命令
print("开始重新打包程序...")
result = subprocess.run(['pyinstaller', 'main.spec'], capture_output=True, text=True)

# 打印输出结果
print("打包输出:")
print(result.stdout)

if result.returncode != 0:
    print("打包失败!")
    print("错误信息:")
    print(result.stderr)
else:
    print("打包成功!")
    print("可执行文件位于dist/main目录下")

input("按Enter键退出...")