# Python bindings for d3d12, dxgi and some windows api for graphics prototyping

```
# Launch powershell with venv
scripts/start.bat
# Install pyd3d12 into the local venv
scripts/install.bat
```

When opening up in vscode it will run scripts/start.bat automatically on each terminal session so only need to run scripts/install.bat to install this project into the virtual environment and build the native modules.

To rebuild native modules you can open up the solution in 'build\temp.win-amd64-cpython-311\Release\_skbuild\PyD3D12.sln' or calle scripts/rebuild_native.bat

Most of the bindings live here at 'native\src\lib.cpp' which is mostly just copilot generated code with some guiding. Objects use *Wrapper classes to wrap methods with python friendly objects, hopefully the API is not to different to the vanilla C++ so that it looks familiar to the people.


To control which local native build type is being loaded at runtime:
```python
def set_build_type(build_type):
    os.environ["NATIVE_BUILD_TYPE"] = build_type

```