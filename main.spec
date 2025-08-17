# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


# 尝试自动获取cv2路径
import cv2
import os
cv2_path = os.path.dirname(cv2.__file__)

a = Analysis(['main.py'],
             pathex=[r'D:\新建文件夹 (2)\cv2', cv2_path],
             binaries=[],
             hiddenimports=['pkg_resources.py2_warn', 'cv2', 'numpy', 'PIL', 'cv2.cv2', 'cv2.data', 'cv2.imgproc', 'cv2.videoio'],
             datas=[('yolov3-tiny.cfg', '.'), ('yolov3-tiny.weights', '.'), ('coco.names', '.'), ('yolov3.cfg', '.'), ('yolov3.weights', '.'), ('MobileNetSSD_deploy.prototxt.txt', '.'), (os.path.join(cv2_path, '*.dll'), '.'), (os.path.join(cv2_path, 'data', '*'), 'data')],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='main',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
