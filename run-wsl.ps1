param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $PipelineArgs
)

$ErrorActionPreference = "Stop"

$projectWindows = [System.IO.Path]::GetFullPath($PSScriptRoot)
if ($projectWindows -notmatch '^([A-Za-z]):(.*)$') {
    throw "The project must be on a Windows drive that is mounted in WSL."
}
$drive = $Matches[1].ToLowerInvariant()
$relativePath = $Matches[2].Replace('\', '/')
$projectLinux = "/mnt/$drive$relativePath"
$wslHome = (& wsl -- sh -lc 'printf %s "$HOME"').Trim()
$pythonLinux = "$wslHome/.venvs/kaggle-s6e8/bin/python"
$ncclLibrary = "$wslHome/.local/nccl-2.28.9-cuda12.9-ubuntu2204/usr/lib/x86_64-linux-gnu"
$runtimeLibraries = "${ncclLibrary}:/usr/local/cuda-12.9/lib64:/usr/lib/wsl/lib"

& wsl -- env "LD_LIBRARY_PATH=$runtimeLibraries" `
    $pythonLinux `
    "$projectLinux/run.py" `
    --config "$projectLinux/configs/gpu_wsl.yaml" `
    @PipelineArgs

exit $LASTEXITCODE
