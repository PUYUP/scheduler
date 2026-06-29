from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "BAAI/bge-m3", 
    model_kwargs={
        "use_safetensors": True,
        "force_download": True # Memaksa sistem mengambil versi terbaru dari server
    }
)