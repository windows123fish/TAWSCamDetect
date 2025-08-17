@echo off

:: 获取当前目录的短路径
for %%i in (.) do set SHORT_PATH=%%~si

:: 构建main.exe的短路径
set MAIN_EXE_PATH=%SHORT_PATH%\dist\main\main.exe

:: 运行main.exe
"%MAIN_EXE_PATH%"
pause