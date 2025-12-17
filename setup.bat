@echo off
REM Script de configuración para el Curso de IA desde Cero
REM ========================================================

echo.
echo =====================================================
echo   CONFIGURACION DEL ENTORNO DE IA
echo =====================================================
echo.

REM Verificar si Python está instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no está instalado o no está en el PATH
    echo Por favor instala Python 3.8 o superior desde python.org
    pause
    exit /b 1
)

echo [OK] Python detectado
python --version

echo.
echo Creando entorno virtual...
python -m venv venv

if %errorlevel% neq 0 (
    echo [ERROR] No se pudo crear el entorno virtual
    pause
    exit /b 1
)

echo [OK] Entorno virtual creado

echo.
echo Activando entorno virtual...
call venv\Scripts\activate.bat

echo.
echo Actualizando pip...
python -m pip install --upgrade pip

echo.
echo Instalando dependencias...
echo (Esto puede tomar varios minutos)
echo.

pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [ADVERTENCIA] Hubo problemas instalando algunas dependencias
    echo Puedes instalarlas manualmente con:
    echo   pip install [nombre-paquete]
    echo.
) else (
    echo.
    echo [OK] Todas las dependencias instaladas correctamente
)

echo.
echo =====================================================
echo   CONFIGURACION COMPLETADA
echo =====================================================
echo.
echo Para activar el entorno en el futuro:
echo   venv\Scripts\activate
echo.
echo Para desactivar el entorno:
echo   deactivate
echo.
echo Para empezar:
echo   cd 01-fundamentos-matematicos
echo   python 01-vectores-matrices.py
echo.
echo =====================================================

pause
