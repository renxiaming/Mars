@echo off
echo 监控训练Loss变化...
echo.

:loop
if exist "C:\Mars_Output\mars_user\c1.nano.full\__cache__\info.txt" (
    echo =========== %TIME% ===========
    type "C:\Mars_Output\mars_user\c1.nano.full\__cache__\info.txt"
    echo.
) else (
    echo 等待训练开始...
)

timeout 30 >nul
goto loop 