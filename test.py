from gettext import install


a=input()
b=input()
c=input()

if a == "--v":
    print("Version 1.0.0")
elif a == "--help":
    print("Usage: python test.py [--v | --help]")
if b =="install":
    print("安装中")
    