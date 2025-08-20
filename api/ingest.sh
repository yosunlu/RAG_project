#!/bin/bash

# 將所有參數組合成一句話
TEXT=$(printf "%s " "$@" | sed 's/ *$//')  # 去掉結尾空格
ENCODED_TEXT=$(python3 -c "import urllib.parse, sys; print(urllib.parse.quote(sys.argv[1]))" "$TEXT")

# 發送 POST 請求
curl -X POST "http://127.0.0.1:8000/ingest" -d "text=$ENCODED_TEXT"