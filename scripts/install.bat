cd %~dp0
cd ..

uv pip uninstall pyd3d12
set SKBUILD_SKIP_BUILD=1
set SKBUILD_SKIP_CMAKE=1
uv pip install -v -e .
python.exe setup.py build