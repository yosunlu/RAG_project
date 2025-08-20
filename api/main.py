from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import uuid
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connect to Milvus
try:
    connections.connect("default", host="localhost", port="19530")
    logger.info("Connected to Milvus successfully")
except Exception as e:
    logger.error(f"Failed to connect to Milvus: {e}")
    raise

# Define collection name
COLLECTION_NAME = "rag_demo"

# Initialize embedder and LLM
try:
    embedder = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model="llama3.1:8b")  # 使用正確的模型名稱
    logger.info("Ollama embedder and LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Ollama models: {e}")
    raise

# Create or load collection
def initialize_collection():
    try:
        if not utility.has_collection(COLLECTION_NAME):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=36),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            ]
            schema = CollectionSchema(fields, description="RAG demo collection")
            collection = Collection(name=COLLECTION_NAME, schema=schema)
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            logger.info(f"Collection '{COLLECTION_NAME}' created with index")
        else:
            collection = Collection(COLLECTION_NAME)
            if not collection.has_index():
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                collection.create_index(field_name="embedding", index_params=index_params)
                logger.info(f"Index recreated for collection '{COLLECTION_NAME}'")
        collection.load()
        logger.info(f"Collection '{COLLECTION_NAME}' loaded successfully")
        return collection
    except Exception as e:
        logger.error(f"Failed to initialize collection: {e}")
        raise

# Initialize collection
collection = initialize_collection()

# Create FastAPI app
app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok"}

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

@app.get("/ask")
def ask(q: str):
    try:
        embedding = embedder.embed_query(q)
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=3,
            output_fields=["content"]
        )
        hits = [hit.entity.get("content") for hit in list(results[0])]
        context = "\n".join(hits)
        prompt = f"""You are a helpful assistant. Use the following context to answer the question concisely and accurately. If the context doesn't provide enough information, say so and provide a general answer.

Context:
{context}

Question:
{q}

Answer:"""
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