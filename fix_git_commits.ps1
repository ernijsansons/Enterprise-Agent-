# Enterprise Agent - Git Commit Fix Script (PowerShell)
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Enterprise Agent - Git Commit Fix Script" -ForegroundColor Cyan  
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/6] Checking Git status..." -ForegroundColor Yellow
git status --porcelain
Write-Host ""

Write-Host "[2/6] Configuring Git settings..." -ForegroundColor Yellow
git config core.autocrlf true
git config core.safecrlf true
Write-Host "Git configuration updated." -ForegroundColor Green
Write-Host ""

Write-Host "[3/6] Adding .gitignore updates..." -ForegroundColor Yellow
git add .gitignore
Write-Host ".gitignore staged." -ForegroundColor Green
Write-Host ""

Write-Host "[4/6] Staging core source code changes..." -ForegroundColor Yellow
git add src/
git add tests/
Write-Host "Source code staged." -ForegroundColor Green
Write-Host ""

Write-Host "[5/6] Staging configuration files..." -ForegroundColor Yellow
git add Makefile
git add poetry.lock
Write-Host "Configuration files staged." -ForegroundColor Green
Write-Host ""

Write-Host "[6/6] Committing all changes..." -ForegroundColor Yellow
$commitMessage = @"
feat: implement comprehensive test fixes and enhancements

- Fix all 18 failing tests with proper mocking and assertions
- Enhance authentication manager with better error handling  
- Add usage monitoring and notification systems
- Implement security audit logging
- Update configuration validation
- Add persistent session management
- Improve error handling across all providers
- Update .gitignore for generated files

Resolves: All test failures, improves system reliability
"@

git commit -m $commitMessage

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ COMMIT SUCCESSFUL!" -ForegroundColor Green
    Write-Host ""
    Write-Host "[7/7] Pushing to remote..." -ForegroundColor Yellow
    git push origin main
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ PUSH SUCCESSFUL!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Final status:" -ForegroundColor Cyan
        git status
    } else {
        Write-Host ""
        Write-Host "❌ PUSH FAILED - Check authentication" -ForegroundColor Red
        Write-Host "Try: git push origin main" -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "❌ COMMIT FAILED" -ForegroundColor Red
    Write-Host "Check the error message above" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Git fix script completed" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
