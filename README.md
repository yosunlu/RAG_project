## RAG Demo (FastAPI + Ollama + Milvus)
### 功能

- 文件寫入：輸入文字，轉換成向量後存進 Milvus。
- 問答查詢：輸入問題，先檢索相關內容，再交給 LLM 生成答案。
- 重置資料：刪除並重建資料庫。

### 實踐方式
- FastAPI：提供 /ingest、/ask、/reset API。
- Ollama：需要先在 地端安裝 Ollama，使用 nomic-embed-text 做向量嵌入，llama3.1:8b 回答問題。
- Milvus：透過 Docker 架設，儲存向量並提供相似度檢索。

### 使用流程

- 安裝並啟動 Ollama（下載模型 nomic-embed-text 與 llama3.1:8b）。
- 用 Docker 啟動 Milvus。
- 啟動 FastAPI server。
- 使用 /ingest API 新增文件。
- 使用 /ask API 提問，回答會根據已經寫入的內容生成。

### Demo:
- https://www.youtube.com/watch?v=lNb5nwaFs74&ab_channel=%E5%91%82%E5%8F%88%E5%B1%B1
