import pydantic
from pydantic import BaseModel
BaseModel.__orig_setstate__ = BaseModel.__setstate__

def _compat_setstate(self, state: dict):
    state.setdefault("__fields_set__", state.get("__pydantic_fields_set__", set()))
    return BaseModel.__orig_setstate__(self, state)

BaseModel.__setstate__ = _compat_setstate

import os
from langchain_community.vectorstores.faiss import FAISS
from langchain import PromptTemplate
from langchain.chains import RetrievalQA, RefineDocumentsChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains.llm import LLMChain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

embed_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

def load_chatbot(index_path: str = None):
    if index_path is None:
        index_path = os.path.join(os.path.dirname(__file__), "faiss_index")

    vectorstore = FAISS.load_local(
        index_path,
        embeddings=embed_model,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6}
    )

    template = """
You are a precise technical assistant.
Use ONLY the provided context to answer the question.
If the context doesn’t cover it, say "I don't know."

Question:
{question}

Context:
{input_documents}

Answer:
"""
    prompt = PromptTemplate(
        template=template,
        input_variables=["question", "input_documents"]
    )

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0,
        top_p=0.7
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    initial_llm_chain = LLMChain(llm=llm, prompt=prompt)
    refine_llm_chain = LLMChain(llm=llm, prompt=prompt)

    refine_chain = RefineDocumentsChain(
        initial_llm_chain=initial_llm_chain,
        refine_llm_chain=refine_llm_chain,
        document_variable_name="input_documents",
        initial_response_name="initial_response"
    )

    qa_chain = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=refine_chain,
        input_key="question"
    )

    return qa_chain

qa_chain = load_chatbot()

def get_chatbot_answer(query: str) -> str:
    return qa_chain.run({"question": query})
