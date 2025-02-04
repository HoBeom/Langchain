from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from langchain.schema import HumanMessage

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

st.title("DocumentGPT")

st.markdown(
    """
Welcome to DocumentGPT!

This chatbot uses GPT-4o-mini to help you find answers from your files.

Simply enter your API key and upload your files using the sidebar to get started.

Happy exploring! 🚀
"""
)

with st.sidebar:
    st.subheader("OpenAI API Key")    
    key = st.text_input("API Key", type="password")
    if "key_test" not in st.session_state:
        st.session_state["key_test"] = False
    if key != "":
        if not st.session_state["key_test"]:
            try:
                llm_test = ChatOpenAI(
                    model="gpt-4o-mini",
                    openai_api_key=key,
                    temperature=0.7,
                )
                llm_test([HumanMessage(content="Test connection.")])
                st.success("✅ API Key is valid!")
                st.session_state["key_test"] = True
            except Exception:
                st.error(f"❌ Invalid API Key")
                st.session_state["key_test"] = False
        else:
            st.success("✅ API Key is valid!")

    st.subheader("Upload File")
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if not st.session_state["key_test"]:
    st.error("⚠️ Please enter a valid OpenAI API Key to proceed.")
    st.stop()

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=key,
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    for chunk in cache_dir.yield_keys():
        cache_dir.mdelete(chunk)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)

else:
    st.session_state["messages"] = []
    st.info("🔑 API Key has been successfully tested!")
    st.info("📁 Upload a file to get started!")
