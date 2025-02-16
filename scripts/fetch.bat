@REM Script to fetch various tools and dependencies

if "%1" == "compressonator" (
    if not exist bin\compressonatorcli.zip (
        curl --tlsv1.2 -kL https://github.com/GPUOpen-Tools/compressonator/releases/download/V4.5.52/compressonatorcli-4.5.52-win64.zip -o bin\compressonatorcli.zip
        powershell -command "Expand-Archive -Path bin\compressonatorcli.zip -DestinationPath bin"
    )
)