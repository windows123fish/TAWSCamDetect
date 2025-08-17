import cv2
import os

# 获取cv2模块的安装路径
cv2_path = os.path.dirname(cv2.__file__)
print(f"cv2模块安装路径: {cv2_path}")

# 获取cv2模块的所有子模块
try:
    import pkgutil
    submodules = [name for _, name, _ in pkgutil.iter_modules([cv2_path])]
    print(f"cv2子模块列表: {submodules}")
except Exception as e:
    print(f"获取子模块时出错: {e}")

# 查找cv2的dll文件
for root, dirs, files in os.walk(cv2_path):
    for file in files:
        if file.endswith('.dll'):
            print(f"找到cv2 dll文件: {os.path.join(root, file)}")

input("按Enter键退出...")