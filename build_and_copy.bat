@echo off
echo [1/3] Сборка exe...
pyinstaller --onefile --add-data "templates;templates" main.py

echo [2/3] Копирование папки models...
xcopy /E /Y /I models dist\models

echo [3/3] Готово! Файл main.exe и модели лежат в dist\
pause