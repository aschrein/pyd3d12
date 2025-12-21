@REM Script to fetch uv and setup a local python env on windows

@REM Fetch uv.exe by url from github releases

cd %~dp0
cd ..
mkdir bin

if not exist bin\uv.exe (
    curl --tlsv1.2 -kL https://github.com/astral-sh/uv/releases/download/0.5.31/uv-x86_64-pc-windows-msvc.zip -o bin/uv.zip
    powershell -command "Expand-Archive -Path bin/uv.zip -DestinationPath bin"
)

set UV_CACHE_DIR=%~dp0..\bin\uv_cache

@REM Add bin to PATH
set PATH=%~dp0..\bin;%PATH%

if not exist bin\python (
    uv.exe --cache-dir %UV_CACHE_DIR% python install 3.11 --install-dir bin\python
)

@REM Setup venv

if not exist .\.venv\Scripts\activate.bat (
    uv.exe --cache-dir %UV_CACHE_DIR% venv --python bin\python\cpython-3.11.11-windows-x86_64-none\python.exe .\.venv
)
call .venv\Scripts\activate.bat

if not exist bin\nuget.exe (
    curl --tlsv1.2 -kL https://dist.nuget.org/win-x86-commandline/latest/nuget.exe -o bin\nuget.exe
)

if not exist bin\Debugviewpp.exe (
    curl --tlsv1.2 -kL https://github.com/CobaltFusion/DebugViewPP/releases/download/v1.9.0.28/Debugviewpp.exe -o bin\Debugviewpp.exe
)

@REM Install Agility SDK for D3D12
if not exist bin\Microsoft.Direct3D.D3D12.1.615.0 (
@REM https://www.nuget.org/packages/Microsoft.Direct3D.D3D12/1.615.0
    cd bin
    nuget.exe install Microsoft.Direct3D.D3D12 -Version 1.615.0
    cd ..
)

@REM Fetch DirectX HLSL compiler
if not exist bin\dxc_2024_07_31 (
    curl --tlsv1.2 -kL https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.8.2407/dxc_2024_07_31.zip -o bin\dxc_2024_07_31.zip
    powershell -command "Expand-Archive -Path bin\dxc_2024_07_31.zip -DestinationPath bin\dxc_2024_07_31"
)

@REM ---------------------------------------
uv add psutil setuptools wheel scikit-build scikit-build-core pybind11 cmake numpy matplotlib
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install cuda-python==12.8.0
uv pip install PyQt5 piq gltflib k3d fastapi uvicorn pydantic
uv pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
@REM ---------------------------------------

powershell