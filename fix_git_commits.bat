@echo off
echo ========================================
echo Enterprise Agent - Git Commit Fix Script
echo ========================================
echo.

echo [1/6] Checking Git status...
git status --porcelain
echo.

echo [2/6] Configuring Git settings...
git config core.autocrlf true
git config core.safecrlf true
echo Git configuration updated.
echo.

echo [3/6] Adding .gitignore updates...
git add .gitignore
echo .gitignore staged.
echo.

echo [4/6] Staging core source code changes...
git add src/
git add tests/
echo Source code staged.
echo.

echo [5/6] Staging configuration files...
git add Makefile
git add poetry.lock
echo Configuration files staged.
echo.

echo [6/6] Committing all changes...
git commit -m "feat: implement comprehensive test fixes and enhancements

- Fix all 18 failing tests with proper mocking and assertions
- Enhance authentication manager with better error handling  
- Add usage monitoring and notification systems
- Implement security audit logging
- Update configuration validation
- Add persistent session management
- Improve error handling across all providers
- Update .gitignore for generated files

Resolves: All test failures, improves system reliability"

if %errorlevel% equ 0 (
    echo.
    echo ✅ COMMIT SUCCESSFUL!
    echo.
    echo [7/7] Pushing to remote...
    git push origin main
    if %errorlevel% equ 0 (
        echo.
        echo ✅ PUSH SUCCESSFUL!
        echo.
        echo Final status:
        git status
    ) else (
        echo.
        echo ❌ PUSH FAILED - Check authentication
        echo Try: git push origin main
    )
) else (
    echo.
    echo ❌ COMMIT FAILED
    echo Check the error message above
)

echo.
echo ========================================
echo Git fix script completed
echo ========================================
pause
