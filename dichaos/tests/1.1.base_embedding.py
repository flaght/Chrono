import sys, os, pdb
from dotenv import load_dotenv

#sys.path.insert(0, os.path.abspath('..'))
load_dotenv()

from dichaos.services.embedding import EmbeddingServiceFactory

text = "this is a test"

## openai
openai_embedding = EmbeddingServiceFactory.create_embedding_service(
    embedding_model='bge-large', embedding_provider='openai', verbose=False)

print(openai_embedding(text))

## huggingface
huggingface_embedding = EmbeddingServiceFactory.create_embedding_service(
    embedding_model='BAAI/bge-small-en',
    embedding_provider='huggingface',
    verbose=False)
print(huggingface_embedding(text))

## azure openai
openai_embedding = EmbeddingServiceFactory.create_embedding_service(
    embedding_model='text-embedding-ada-002',
    embedding_provider='azureopenai',
    chunk_size=5000,
    verbose=False)

print(openai_embedding(text))
