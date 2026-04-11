param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if ($MyInvocation.InvocationName -eq ".") {
    Write-Warning "Ce script est preferablement lance avec '& .\Launch-ProductIntelligence-Server-And-Ngrok.ps1'."
}

function Resolve-PythonRuntime {
    $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return @{
            Path = $venvPython
            PrefixArgs = @()
            Source = ".venv"
        }
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        Write-Warning "Python de .venv introuvable. Utilisation de python depuis le PATH."
        return @{
            Path = $pythonCmd.Source
            PrefixArgs = @()
            Source = "python"
        }
    }

    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        Write-Warning "Python de .venv introuvable. Utilisation du launcher 'py -3'."
        return @{
            Path = $pyCmd.Source
            PrefixArgs = @("-3")
            Source = "py"
        }
    }

    return $null
}

$pythonRuntime = Resolve-PythonRuntime
if (-not $pythonRuntime) {
    Write-Host "Aucun runtime Python n'a ete trouve. Installe Python puis relance le script."
    return
}

$pythonPath = $pythonRuntime.Path
$pythonPrefixArgs = $pythonRuntime.PrefixArgs

Write-Host "[1/2] Lancement du serveur Product Intelligence sur http://$BindHost`:$Port ..."
Start-Process -FilePath $pythonPath -ArgumentList @($pythonPrefixArgs + @("-m", "uvicorn", "app.main:app", "--host", $BindHost, "--port", "$Port", "--reload")) -WorkingDirectory $projectRoot | Out-Null

Write-Host "[2/2] Lancement de ngrok (http $Port) ..."
Write-Host "La fenetre ngrok affichera l'URL publique a utiliser."

$ngrokCmd = Get-Command ngrok -ErrorAction SilentlyContinue
if ($ngrokCmd) {
    & $ngrokCmd.Source http $Port
    return
}

& $pythonPath @pythonPrefixArgs -c "import pyngrok" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Warning "Commande ngrok absente du PATH. Utilisation de pyngrok via Python."
    & $pythonPath @pythonPrefixArgs -m pyngrok ngrok http $Port
    return
}

Write-Host "ngrok est introuvable." 
Write-Host "Option 1 (recommandee): winget install ngrok.ngrok"
Write-Host "Option 2 (fallback): $pythonPath $($pythonPrefixArgs -join ' ') -m pip install pyngrok"
Write-Host "Puis relance ce script."
