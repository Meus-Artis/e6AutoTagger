$gpus = Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name | grep AMD
if (-not $gpus) {
    Write-Error "No AMD GPU detected."
    exit 1
}
$gpuString = ($gpus -join " ").ToLower()
$arch = $null
switch -Regex ($gpuString) {
    "MI25$"               { $arch = "900"; break }
    "(MI(5|6)|VI)"        { $arch = "906"; break }
    "MI1"                 { $arch = "908"; break }
    "MI2(1|5.)"           { $arch = "90a"; break }
    "MI3(2|0)"            { $arch = "942"; break }
    "MI35"                { $arch = "950"; break }
    "RX\s5"               { $arch = "1010"; break }
    "V5"                  { $arch = "1011"; break }
    "W5"                  { $arch = "1012"; break }
    "RX\s6(9|8)|(W68|V6)" { $arch = "1030"; break }
    "RX\s67"              { $arch = "1031"; break }
    "(RX\s|W)66"          { $arch = "1032"; break }
    "Gogh"                { $arch = "1033"; break }
    "RX\s65"              { $arch = "1034"; break }
    "680M"                { $arch = "1035"; break }
    "Raphael"             { $arch = "1036"; break }
    "RX\s79|W7(8|9)"      { $arch = "1100"; break }
    "(RX\s7(8|7)|V7|W77)" { $arch = "1101"; break }
    "RX\s76"              { $arch = "1102"; break }
    "Ryzen (7|9)"         { $arch = "1103"; break }
    "AI 9"                { $arch = "1150"; break }
    "AI M"                { $arch = "1151"; break }
    "AI 7"                { $arch = "1152"; break }
    "820M"                { $arch = "1153"; break }
    "RX\s906"             { $arch = "1200"; break }
    "(RX\s907)|R9(7|6)"   { $arch = "1201"; break }
}
if (-not $arch) {
    Write-Error "Not a supported AMD GPU: $gpuString"
    exit 1
}
$cmd = "venv\Scripts\pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ torch[device-gfx$arch]"
Invoke-Expression $cmd