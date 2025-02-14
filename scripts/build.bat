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

if not exist venv\Scripts\activate.bat (
    uv.exe --cache-dir %UV_CACHE_DIR% venv --python bin\python\cpython-3.11.11-windows-x86_64-none\python.exe ./venv
)

@REM ---------------------------------------
uv pip install setuptools wheel scikit-build scikit-build-core pybind11
uv pip install cmake
@REM ---------------------------------------

call venv\Scripts\activate.bat

uv pip uninstall pyd3d12
set SKBUILD_SKIP_BUILD=1
set SKBUILD_SKIP_CMAKE=1
uv pip install -v -e .
python.exe setup.py build

powershell