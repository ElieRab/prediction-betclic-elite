@echo off
REM Lance l'interface web puis ouvre le navigateur.
cd /d "%~dp0"
start "" http://127.0.0.1:8010
python -m bepred web --port 8010
