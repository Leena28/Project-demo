# # chatbot/rag_chatbot.py
# from langchain.chains import RetrievalQA
# #from langchain.vectorstores import FAISS
# #from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain.llms import HuggingFacePipeline
# import torch

# # Load FAISS index and embeddings
# def load_chatbot():
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vectorstore = FAISS.load_local(
#         "C:/Users/CG-DTE/Desktop/Project/startup_app/chatbot/faiss_index",
#         embeddings,
#         allow_dangerous_deserialization=True
#     )

#     #model_id = "microsoft/phi-1_5"
#     model_id="google/flan-t5-small"
#     tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir="./model_cache")
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         torch_dtype=torch.float32,
#         device_map=None
#     ).to("cuda" if torch.cuda.is_available() else "cpu")

#     gen_pipeline = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=128,
#         do_sample=False
#     )
#     llm = HuggingFacePipeline(pipeline=gen_pipeline)

#     retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         chain_type="stuff",
#         return_source_documents=False
#     )
#     return qa_chain

# # Global object
# qa_chain = load_chatbot()

# # Predict
# def get_chatbot_answer(query: str) -> str:
#     ##result = qa_chain.run(query)
#     result = qa_chain.invoke({"query": query})

#     print("LLM responded:",result)
#     return result

# chatbot/rag_chatbot.py

#BLOCK 2 ACTUAL CODE 2
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Suppress Windows symlink warning

from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline  # ✅ FIXED HERE
from langchain.llms import HuggingFacePipeline
import torch

# Load FAISS index and embeddings
def load_chatbot():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        "C:/Users/CG-DTE/Desktop/Project/startup_app/chatbot/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    model_id = "google/flan-t5-small"  # ✅ Using T5-compatible model
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./model_cache")
    model = AutoModelForSeq2SeqLM.from_pretrained(  # ✅ FIXED HERE
        model_id,
        torch_dtype=torch.float32,
        device_map=None
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    gen_pipeline = pipeline(
        "text2text-generation",  # ✅ For seq2seq models like T5
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        do_sample=False
    )
    llm = HuggingFacePipeline(pipeline=gen_pipeline)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False
    )
    return qa_chain

# Global object
qa_chain = load_chatbot()

# Predict
def get_chatbot_answer(query: str) -> str:
    result = qa_chain.invoke({"query": query})
    print("LLM responded:", result)
    return result

##ENDOF ACTUAL CODE 2

##ACTUAL CODE 3---
# # chatbot/rag_chatbot.py

# import os
# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Suppress Windows symlink warning

# from langchain.chains import RetrievalQA
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain.llms import HuggingFacePipeline
# import torch

# def load_chatbot():
#     # Load vector index and embedding model
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vectorstore = FAISS.load_local(
#         "C:/Users/CG-DTE/Desktop/Project/startup_app/chatbot/faiss_index",
#         embeddings,
#         allow_dangerous_deserialization=True
#     )

#     # Use Microsoft Phi-1.5 model for text generation (on CPU)
#     # model_id = "microsoft/phi-1_5"
#     # tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./model_cache")
#     # model = AutoModelForCausalLM.from_pretrained(
#     #     model_id,
#     #     torch_dtype=torch.float32,
#     #     device_map="cpu",  # Ensures CPU-only
#     #     cache_dir="./model_cache"
#     # ).to("cpu")  # Explicitly move to CPU

#     model_id = "microsoft/phi-1_5"
#     tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./model_cache")

#     model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.float32,
#     cache_dir="./model_cache"
#     ).to("cpu")  # ✅ Don't use device_map


#     # Generation pipeline
#     gen_pipeline = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=128,
#         do_sample=False
#     )
#     llm = HuggingFacePipeline(pipeline=gen_pipeline)

#     # QA Chain
#     retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         chain_type="stuff",
#         return_source_documents=False
#     )
#     return qa_chain

# # Initialize once globally
# qa_chain = load_chatbot()

# # Predict function
# def get_chatbot_answer(query: str) -> str:
#     result = qa_chain.invoke({"query": query})
#     print("LLM responded:", result)
#     return result
