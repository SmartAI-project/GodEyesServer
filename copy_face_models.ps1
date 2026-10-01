$ErrorActionPreference = "Stop"

$src = "D:\GodEyesServer\data\face_models"
$dst = Join-Path $PSScriptRoot "assets\models"

New-Item -ItemType Directory -Force -Path $dst | Out-Null

$yunet = Join-Path $src "face_detection_yunet_2023mar.onnx"
$sface = Join-Path $src "face_recognition_sface_2021dec_int8.onnx"

if (-not (Test-Path $yunet)) {
    throw "Missing YuNet model: $yunet"
}
if (-not (Test-Path $sface)) {
    throw "Missing SFace model: $sface"
}

Copy-Item $yunet (Join-Path $dst "face_detection_yunet_2023mar.onnx") -Force
Copy-Item $sface (Join-Path $dst "face_recognition_sface_2021dec_int8.onnx") -Force

Write-Host "Face ID models copied successfully." -ForegroundColor Green
Write-Host "Destination: $dst"
