param(
    [ValidateSet('home', 'capture', 'settlement', 'reconstruction', 'legacy', 'videos')]
    [string]$Mode = 'home',
    [switch]$SetupOnly,
    [switch]$Repair,
    [switch]$Offline
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:PYTHONNOUSERSITE = '1'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runtimeRoot = Join-Path $projectRoot '.runtime'
$venvRoot = Join-Path $projectRoot '.venv'
$pythonExe = Join-Path $venvRoot 'Scripts\python.exe'
$runtimeExe = Join-Path $runtimeRoot 'python\python.exe'
$logRoot = Join-Path $projectRoot 'logs'
$runtimeLock = $null
$usageLock = $null
$environmentInUse = $false
$transcribing = $false
$exitCode = 0

function Test-Python([string]$Executable, [switch]$IsVenv) {
    if (-not (Test-Path -LiteralPath $Executable)) { return $false }
    $probe = 'import sys,struct,ssl,venv,ensurepip; from pathlib import Path; assert sys.version_info[:2]==(3,12) and struct.calcsize(chr(80))==8; '
    if ($IsVenv) {
        $probe += 'assert sys.prefix!=sys.base_prefix; assert Path(sys._base_executable).resolve()==Path(sys.argv[1]).resolve()'
    }
    try {
        $null = & $Executable -I -c $probe $runtimeExe 2>&1
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

function Backup-Environment([string]$Path) {
    $resolved = [IO.Path]::GetFullPath($Path)
    $allowed = @($venvRoot, (Join-Path $runtimeRoot 'python'))
    if ($resolved -notin $allowed) { throw "Unexpected environment path: $resolved" }
    if (Test-Path -LiteralPath $resolved) {
        if ((Get-Item -LiteralPath $resolved).Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Environment is a link. Move it manually before retrying: $resolved"
        }
        $backupRoot = Join-Path $runtimeRoot 'backups'
        New-Item -ItemType Directory -Force -Path $backupRoot | Out-Null
        $backup = Join-Path $backupRoot ((Split-Path $resolved -Leaf) + '-' + [guid]::NewGuid().ToString('N'))
        if (-not ([IO.Path]::GetFullPath($backup).StartsWith($runtimeRoot + '\', [StringComparison]::OrdinalIgnoreCase))) {
            throw 'Backup path is outside the runtime directory.'
        }
        Write-Host "Preserving the previous environment: $backup"
        Move-Item -LiteralPath $resolved -Destination $backup
    }
}

try {
    if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64') {
        throw '本程序适用于 Intel/AMD 64 位电脑上的 Windows 10/11。'
    }
    New-Item -ItemType Directory -Force -Path $runtimeRoot, $logRoot | Out-Null
    try {
        $runtimeLock = [IO.File]::Open((Join-Path $runtimeRoot 'setup.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    } catch {
        throw '另一个启动窗口正在准备环境，请等待完成后重试。'
    }
    Start-Transcript -Path (Join-Path $logRoot 'bootstrap.log') -Append | Out-Null
    $transcribing = $true
    try {
        $usageLock = [IO.File]::Open((Join-Path $runtimeRoot 'in_use.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    } catch { $environmentInUse = $true }
    if ($Repair -and $environmentInUse) { throw '请先关闭所有采集、处理和工作台窗口，再修复环境。' }
    Write-Host 'D435i 数采软件 v1.0.0 — 正在准备运行环境' -ForegroundColor Cyan
    Write-Host "软件位置：$projectRoot"
    $manifest = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $PSScriptRoot 'runtime.json') | ConvertFrom-Json
    $runtimeStamp = Join-Path $runtimeRoot 'python\.runtime.sha256'
    $completeRuntime = (Test-Path -LiteralPath $runtimeStamp) -and ((Get-Content -Raw -LiteralPath $runtimeStamp).Trim() -eq $manifest.sha256)
    if (-not $completeRuntime -or -not (Test-Python $runtimeExe)) {
        if ($environmentInUse) { throw '请先关闭所有软件窗口，再恢复 Python 运行环境。' }
        Write-Host '[1/3] 准备软件专用 Python（首次约需下载 30 MB）…'
        $downloadRoot = Join-Path $runtimeRoot 'downloads'
        New-Item -ItemType Directory -Force -Path $downloadRoot | Out-Null
        $archive = Join-Path $downloadRoot $manifest.filename
        $offlineArchive = Join-Path (Join-Path $projectRoot 'offline') $manifest.filename
        if (Test-Path -LiteralPath $offlineArchive) { Copy-Item -LiteralPath $offlineArchive -Destination $archive }
        $validArchive = (Test-Path -LiteralPath $archive) -and ((Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash -eq $manifest.sha256)
        if (-not $validArchive) {
            if ($Offline) { throw '离线 Python 运行包缺失或损坏，请使用完整离线包。' }
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            $partial = $archive + '.partial'
            $downloaded = $false
            for ($attempt = 1; $attempt -le 3; $attempt++) {
                try {
                    Invoke-WebRequest -UseBasicParsing -Uri $manifest.url -OutFile $partial -TimeoutSec 180
                    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $partial).Hash -ne $manifest.sha256) { throw 'Python archive SHA256 mismatch.' }
                    Move-Item -LiteralPath $partial -Destination $archive -Force
                    $downloaded = $true
                    break
                } catch {
                    Write-Host "Download attempt $attempt failed: $($_.Exception.Message)"
                }
            }
            if (-not $downloaded) { throw 'Python 下载失败，请检查网络或使用完整离线包，再重新双击启动入口。' }
        }
        Backup-Environment (Join-Path $runtimeRoot 'python')
        Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $runtimeRoot 'python') -Force
        if (-not (Test-Python $runtimeExe)) { throw '软件专用 Python 未能启动，请重新双击入口重试。' }
        Set-Content -NoNewline -Encoding ASCII -LiteralPath $runtimeStamp -Value $manifest.sha256
    } else { Write-Host '[1/3] 软件专用 Python 已就绪。' }

    if (-not (Test-Python $pythonExe -IsVenv)) {
        if ($environmentInUse) { throw '请先关闭所有软件窗口，再重新创建环境。' }
        Write-Host '[2/3] 创建独立依赖环境…'
        Backup-Environment $venvRoot
        & $runtimeExe -I -X utf8 -m venv $venvRoot
        if ($LASTEXITCODE -ne 0) { throw '无法创建独立环境，请检查文件夹写入权限并重试。' }
    }
    $requirements = Join-Path $projectRoot 'requirements.txt'
    $fingerprint = (Get-FileHash -Algorithm SHA256 -LiteralPath $requirements).Hash
    $stamp = Join-Path $venvRoot '.requirements.sha256'
    $probeScript = Join-Path $PSScriptRoot 'environment_check.py'
    $probeOutput = & $pythonExe $probeScript --quick 2>&1
    $probePassed = ($LASTEXITCODE -eq 0)
    $sameRequirements = (Test-Path -LiteralPath $stamp) -and ((Get-Content -Raw -LiteralPath $stamp).Trim() -eq $fingerprint)
    if ($Repair -or -not $probePassed -or -not $sameRequirements) {
        if ($environmentInUse) { throw '请先关闭所有软件窗口，再更新依赖。' }
        Write-Host '[2/3] 安装已锁定版本的依赖，首次运行可能需要几分钟…'
        $pipArgs = @('-I', '-X', 'utf8', '-m', 'pip', '--isolated', 'install', '--disable-pip-version-check', '--only-binary=:all:', '--timeout', '45', '--retries', '2', '-r', $requirements)
        if ($Repair -or -not $probePassed) { $pipArgs += '--force-reinstall' }
        $wheels = Join-Path $projectRoot 'offline\wheels'
        if (Test-Path -LiteralPath $wheels) {
            Write-Host '优先使用随包提供的离线依赖。'
            & $pythonExe @pipArgs --no-index --find-links $wheels
            $installed = ($LASTEXITCODE -eq 0)
        } else { $installed = $false }
        if (-not $installed) {
            if ($Offline) { throw '离线依赖不完整，请使用完整离线包。' }
            & $pythonExe @pipArgs --index-url 'https://pypi.org/simple'
            $installed = ($LASTEXITCODE -eq 0)
        }
        if (-not $installed) { throw '依赖安装失败，请检查网络或使用完整离线包后重试。' }
        & $pythonExe -I -X utf8 -m pip --isolated check
        if ($LASTEXITCODE -ne 0) { throw '依赖兼容性检查未通过，请关闭所有软件窗口后运行 start.bat -Repair。' }
        $verificationOutput = & $pythonExe $probeScript --quick 2>&1
        if ($LASTEXITCODE -ne 0) {
            $verificationOutput | ForEach-Object { Write-Host $_ }
            throw '环境检查未通过，请关闭所有软件窗口后运行 start.bat -Repair。'
        }
        Set-Content -NoNewline -Encoding ASCII -LiteralPath $stamp -Value $fingerprint
    } else { Write-Host '[2/3] 依赖检查通过，无需重新安装。' }
    New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot 'sessions'), (Join-Path $projectRoot 'telemetry') | Out-Null
    Write-Host '[3/3] 环境已就绪。' -ForegroundColor Green
} catch {
    Write-Host ("启动失败：" + $_.Exception.Message) -ForegroundColor Red
    Write-Host "日志：$logRoot\bootstrap.log"
    $exitCode = 1
} finally {
    if ($transcribing) { Stop-Transcript | Out-Null }
    if ($runtimeLock) { $runtimeLock.Dispose() }
    if ($usageLock) { $usageLock.Dispose() }
}

if ($exitCode -eq 0 -and -not $SetupOnly) {
    Set-Location -LiteralPath $projectRoot
    & $pythonExe (Join-Path $projectRoot 'launcher.py') --mode $Mode
    $exitCode = $LASTEXITCODE
}
exit $exitCode
