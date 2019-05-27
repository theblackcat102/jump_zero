rmdir /s /q "D:\Google Drive\code\jump_zero\pure_mcts\build"
rmdir /s /q "D:\Google Drive\code\jump_zero\pure_mcts\dist"
del Team_5.spec
pyinstaller Team_5.py --onefile
AI_game_5s.exe < CMD_PARAM.txt