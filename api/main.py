from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import uuid
import logging

# 設定 logging 等級與格式
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 連線到本地的 Milvus 向量資料庫
try:
    connections.connect("default", host="localhost", port="19530")
    logger.info("Connected to Milvus successfully")
except Exception as e:
    logger.error(f"Failed to connect to Milvus: {e}")
    raise

# 向量集合名稱
COLLECTION_NAME = "rag_demo"

# 初始化地端 embedding 模型與對話 LLM 模型（都透過 Ollama）
try:
    embedder = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model="llama3.1:8b")
    logger.info("Ollama embedder and LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Ollama models: {e}")
    raise

# 初始化或載入 Milvus 向量集合
def initialize_collection():
    try:
        if not utility.has_collection(COLLECTION_NAME):
            # 定義欄位 schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=36),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            ]
            schema = CollectionSchema(fields, description="RAG demo collection")
            collection = Collection(name=COLLECTION_NAME, schema=schema)

            # 建立向量索引（使用 cosine 相似度）
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            logger.info(f"Collection '{COLLECTION_NAME}' created with index")
        else:
            # 如果已存在集合，直接載入
            collection = Collection(COLLECTION_NAME)
            if not collection.has_index():
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                collection.create_index(field_name="embedding", index_params=index_params)
                logger.info(f"Index recreated for collection '{COLLECTION_NAME}'")

        # 將集合載入記憶體供查詢使用
        collection.load()
        logger.info(f"Collection '{COLLECTION_NAME}' loaded successfully")
        return collection
    except Exception as e:
        logger.error(f"Failed to initialize collection: {e}")
        raise

# 初始化集合
collection = initialize_collection()

# 建立 FastAPI 應用程式
app = FastAPI()

# 健康檢查用 endpoint
@app.get("/")
def health():
    return {"status": "ok"}

# 將文字內容轉為向量並儲存至 Milvus
@app.post("/ingest")
def ingest(text: str = Form(...)):
    try:
        embedding = embedder.embed_query(text)
        doc_id = str(uuid.uuid4())
        collection.insert([[doc_id], [text], [embedding]])
        logger.info(f"Ingested document with ID: {doc_id}")
        return {"status": "ingested", "id": doc_id}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# 查詢文字，並透過 Milvus 找到相關內容後用 LLM 回答
@app.get("/ask")
def ask(q: str):
    try:
        embedding = embedder.embed_query(q)
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        # 以向量搜尋相似的內容
        results = collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=3,
            output_fields=["content"]
        )

        # 取出搜尋結果的文字內容
        hits = [hit.entity.get("content") for hit in list(results[0])]
        context = "\n".join(hits)

        # 組合 prompt，餵給 LLM
        prompt = f"""You are a helpful assistant. Use the following context to answer the question concisely and accurately. If the context doesn't provide enough information, say so and provide a general answer.

Context:
{context}

Question:
{q}

Answer:"""

        # 使用 Ollama 執行 LLM 推論
        answer = llm.invoke([HumanMessage(content=prompt)]).content
        logger.info(f"Generated answer for question: {q}")

        return {
            "question": q,
            "related": hits,
            "answer": answer
        }
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# 重置整個向量集合（清空資料）
@app.post("/reset")
def reset_collection():
    try:
        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME)
            logger.info(f"Collection '{COLLECTION_NAME}' dropped")
            global collection
            collection = initialize_collection()
            return JSONResponse(content={"status": "collection dropped and recreated"})
        else:
            return JSONResponse(content={"status": "collection not found"})
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})