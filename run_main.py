import os
import subprocess

# 设置main.exe的完整路径
main_exe_path = r'd:\新建文件夹 (2)\cv2\dist\main\main.exe'

# 使用subprocess运行main.exe
try:
    subprocess.run([main_exe_path], check=True)
except subprocess.CalledProcessError as e:
    print(f'运行失败，错误码: {e.returncode}')
    input('按Enter键退出...')
except Exception as e:
    print(f'发生错误: {e}')
    input('按Enter键退出...')
else:
    print('程序正常退出')
    input('按Enter键退出...')