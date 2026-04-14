$gpus = Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name
if (-not $gpus) {
    Write-Error "No GPU detected."
    exit 1
}
$gpuString = ($gpus -join " ").ToLower()
$arch = $null
switch -Regex ($gpuString) {
    "mi30"   { $arch = "gfx94X-dcgpu"; break }
    "mi35"   { $arch = "gfx950-dcgpu"; break }
    "rx\s*7" { $arch = "gfx110X-all"; break }
    "strix"  { $arch = "gfx1151"; break }
    "rx\s*9" { $arch = "gfx120X-all"; break }
}
if (-not $arch) {
    Write-Error "Not a supported AMD GPU: $gpuString"
    exit 1
}
$indexUrl = "https://rocm.nightlies.amd.com/v2/$arch/"
$cmd = "venv\Scripts\pip install --index-url $indexUrl --pre torch"
Invoke-Expression $cmd