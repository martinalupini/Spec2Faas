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
    await add_entry("FUNCTION MergeSort(Array A, index p, index r):\n    IF p < r THEN:\n        // Find the midpoint of the array\n        q = floor((p + r) / 2)\n\n        // Recursively sort the first half\n        MergeSort(A, p, q)\n\n        // Recursively sort the second half\n        MergeSort(A, q + 1, r)\n\n        // Merge the two sorted halves\n        Merge(A, p, q, r)\n\n// Function that merges two sorted sub-arrays: A[p..q] and A[q+1..r]\nFUNCTION Merge(Array A, index p, index q, index r):\n    // Calculate the sizes of the two sub-arrays\n    n1 = q - p + 1\n    n2 = r - q\n\n    // Create two temporary arrays L (left) and R (right)\n    Create Array L[1..n1+1]\n    Create Array R[1..n2+1]\n\n    // Copy data into the left sub-array L\n    FOR i FROM 1 TO n1 DO:\n        L[i] = A[p + i - 1]\n\n    // Copy data into the right sub-array R\n    FOR j FROM 1 TO n2 DO:\n        R[j] = A[q + j]\n\n    // Add sentinels (infinity or very large values) at the end of L and R\n    L[n1 + 1] = INFINITY\n    R[n2 + 1] = INFINITY\n\n    // Initialize indices for L and R\n    i = 1\n    j = 1\n\n    // Merge L and R back into the main array A[p..r]\n    FOR k FROM p TO r DO:\n        IF L[i] <= R[j] THEN:\n            A[k] = L[i]\n            i = i + 1\n        ELSE:\n            A[k] = R[j]\n            j = j + 1")


if __name__ == "__main__":
    asyncio.run(main())