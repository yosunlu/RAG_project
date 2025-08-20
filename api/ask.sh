#!/bin/bash

# 將所有輸入參數組合成一個查詢字串
QUERY=$(printf "%s " "$@" | sed 's/ *$//')  # 去掉結尾空格
ENCODED_QUERY=$(python3 -c "import urllib.parse, sys; print(urllib.parse.quote(sys.argv[1]))" "$QUERY")

curl "http://127.0.0.1:8000/ask?q=$ENCODED_QUERY"