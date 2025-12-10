import os
import asyncio
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig, \
    SentenceTransformerEmbeddingFunctionConfig, OpenAIEmbeddingFunctionConfig
from Utils import load_env_variables

async def add_entry(function):
    chroma_user_memory = ChromaDBVectorMemory(
        config=PersistentChromaDBVectorMemoryConfig(
            collection_name="functions",
            persistence_path=os.getenv("PERSISTENCE_PATH"),  # Use the temp directory here
            k=1,  # Return top k results
            score_threshold=0.7,  # Minimum similarity score
            embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(
                model_name="all-MiniLM-L6-v2"  # Use default model for testing
            ),
        )
    )

    await chroma_user_memory.add(
        MemoryContent(
            content=function,
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "functions", "type": "function"},
        )
    )

    await chroma_user_memory.close()
    chroma_user_memory.dump_component().model_dump_json()


async def main():
    load_env_variables()
    await add_entry("def count_total_number_labubu()\n    return 1000")
    await add_entry("def foo()\n    return 100")


if __name__ == "__main__":
    asyncio.run(main())