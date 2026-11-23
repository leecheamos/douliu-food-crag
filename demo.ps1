# 斗六美食 LINE demo：一鍵啟動
# 用法（在專案資料夾）：powershell -ExecutionPolicy Bypass -File .\demo.ps1
# 停止：docker compose down
#
# 步驟：啟動 Docker（推薦程式＋Cloudflare 臨時通道）→ 取得新網址 → 確認網址連得到
#       → 用 LINE API 更新 Webhook URL → 用 LINE API 測試連線。
# 金鑰只從 .env 讀取，不會顯示在畫面上。

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Set-Location -Path $PSScriptRoot

function Write-Step($text) { Write-Host "`n▶ $text" -ForegroundColor Cyan }
function Stop-Demo($text) { Write-Host "`n✖ $text" -ForegroundColor Red; exit 1 }

# 1. 前置檢查
Write-Step "檢查 Docker 與 .env"
docker info --format "{{.ServerVersion}}" *> $null
if ($LASTEXITCODE -ne 0) { Stop-Demo "Docker 沒有在執行，請先開啟 Docker Desktop。" }
if (-not (Test-Path ".env")) { Stop-Demo "找不到 .env。" }
$envValues = @{}
foreach ($line in Get-Content ".env" -Encoding UTF8) {
    if ($line -match '^\s*([A-Z_]+)\s*=\s*(.*)$') { $envValues[$Matches[1]] = $Matches[2].Trim().Trim('"').Trim("'") }
}
foreach ($key in "LINE_CHANNEL_SECRET", "LINE_CHANNEL_ACCESS_TOKEN", "GEMINI_API_KEY") {
    if (-not $envValues[$key]) { Stop-Demo ".env 的 $key 是空的。" }
}
$lineHeaders = @{ Authorization = "Bearer " + $envValues["LINE_CHANNEL_ACCESS_TOKEN"] }

# 2. 啟動容器
Write-Step "啟動 Docker（第一次會比較久）"
docker compose up -d --build app
if ($LASTEXITCODE -ne 0) { Stop-Demo "docker compose 啟動失敗，請把上面的錯誤訊息貼給 Claude。" }
# 臨時通道可能已被 Cloudflare 收回（日誌出現 Tunnel not found），所以每次都重建通道、拿新網址。
$since = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
docker compose up -d --force-recreate tunnel
if ($LASTEXITCODE -ne 0) { Stop-Demo "通道啟動失敗，請把上面的錯誤訊息貼給 Claude。" }

# 3. 取得這次新建通道的網址（只看重建之後的日誌）
Write-Step "等待 Cloudflare 通道網址"
$url = $null
for ($i = 0; $i -lt 45 -and -not $url; $i++) {
    $logs = (docker compose logs --since $since tunnel 2>&1) -join "`n"
    $found = [regex]::Matches($logs, 'https://[a-z0-9-]+\.trycloudflare\.com')
    if ($found.Count -gt 0) { $url = $found[$found.Count - 1].Value } else { Start-Sleep -Seconds 2 }
}
if (-not $url) { Stop-Demo "90 秒內沒有拿到網址。請執行 docker compose logs tunnel，把輸出貼給 Claude。" }
Write-Host "  通道網址：$url"

# 4. 確認網址真的連得到程式（新網址的 DNS 可能要幾秒才生效）
Write-Step "確認網址連得到推薦程式"
$health = $null
for ($i = 0; $i -lt 30 -and -not $health; $i++) {
    try { $health = Invoke-RestMethod -Uri "$url/health" -TimeoutSec 10 } catch { Start-Sleep -Seconds 2 }
}
if (-not $health) { Stop-Demo "網址連不到程式。請執行 docker compose logs，把輸出貼給 Claude。" }
if (-not $health.line_configured) { Stop-Demo "程式沒有讀到 LINE 金鑰，請檢查 .env。" }
Write-Host "  健康檢查 OK（embedding：$($health.embedding)）"

# 5. 更新 LINE Webhook URL
$webhook = "$url/webhooks/line"
Write-Step "更新 LINE Webhook URL"
$body = @{ endpoint = $webhook } | ConvertTo-Json
try {
    Invoke-RestMethod -Method Put -Uri "https://api.line.me/v2/bot/channel/webhook/endpoint" `
        -Headers $lineHeaders -ContentType "application/json" -Body $body | Out-Null
} catch { Stop-Demo "LINE 拒絕更新 Webhook（$($_.Exception.Response.StatusCode.value__)），請確認 Access token。" }
$current = Invoke-RestMethod -Uri "https://api.line.me/v2/bot/channel/webhook/endpoint" -Headers $lineHeaders
Write-Host "  已設定：$($current.endpoint)"
if (-not $current.active) {
    Write-Host "  ⚠ LINE 後台的「Use webhook」是關閉的，請到 LINE Developers → Messaging API 打開它。" -ForegroundColor Yellow
}

# 6. 讓 LINE 實際送一次測試（等同後台的 Verify）
Write-Step "請 LINE 測試連線"
$test = Invoke-RestMethod -Method Post -Uri "https://api.line.me/v2/bot/channel/webhook/test" `
    -Headers $lineHeaders -ContentType "application/json" -Body $body
if (-not $test.success) { Stop-Demo "LINE 測試失敗：$($test.statusCode) $($test.reason)" }
Write-Host "  LINE 測試成功（HTTP $($test.statusCode)）"

Write-Host "`n✔ 可以開始測試或錄影了！用手機傳訊息給 LINE 機器人。" -ForegroundColor Green
Write-Host "  查看回覆紀錄（含每次延遲秒數）：docker compose logs -f app"
Write-Host "  結束：docker compose down（網址會失效，下次執行本腳本會自動換新）"
