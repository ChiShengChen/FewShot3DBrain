#!/bin/bash
# 重寫 Git 歷史，排除大檔案後再 push
# 使用前請確保 .gitignore 已正確設定

set -e
cd "$(dirname "$0")/.."

echo "=== 步驟 1: 建立不含歷史的乾淨分支 ==="
git checkout --orphan fresh_main

echo "=== 步驟 2: 加入檔案（依 .gitignore 排除大檔）==="
git add .
git status
echo ""
read -p "請確認上方沒有 data/、weights/、outputs/ 等大目錄。繼續? [y/N] " -r
[[ $REPLY =~ ^[Yy]$ ]] || exit 1

echo "=== 步驟 3: 初次 commit ==="
git commit -m "Initial commit (exclude large data/weights/outputs)"

echo "=== 步驟 4: 替換 main 分支 ==="
git branch -D main 2>/dev/null || true
git branch -m main

echo "=== 步驟 5: 推送（會覆寫遠端）==="
git push -u origin main --force

echo ""
echo "完成！可用 du -sh .git 檢查 .git 目錄大小。"
