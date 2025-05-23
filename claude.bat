@echo off
REM Claude Code Docker Wrapper
REM Save this as claude.bat in your project directory

REM Get the current directory and convert to Unix-style path for Docker
set "CURRENT_DIR=%cd%"
set "UNIX_PATH=%CURRENT_DIR:\=/%"
set "UNIX_PATH=%UNIX_PATH:C:=/c%"

REM Run Claude Code in Docker with current directory mounted
docker run --rm -it ^
  -v "%CURRENT_DIR%:/workspace" ^
  -v "%USERPROFILE%\.claude:/root/.claude" ^
  claude-code %*

REM Note: The second volume mount preserves Claude authentication between runs