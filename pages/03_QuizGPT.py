import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
import regex as re
from langchain.retrievers import WikipediaRetriever
from langchain.schema import HumanMessage

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


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
    docs = None
    topic = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "Wikipedia Article",
            "File",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)
    st.markdown("---")
    model_select = st.selectbox("Model", ("gpt-4o-mini", "gpt-3.5-turbo-1106"))
    level = st.selectbox("Quiz Level", ("EASY", "HRAD"))
    st.markdown("---")
    st.write("Github: https://github.com/hobeom/langchain")
    

if not docs:
    st.markdown(
        """
        ### Welcome to QuizGPT 🎓
        Create personalized quizzes from **Wikipedia articles** or **your uploaded files** to test your knowledge and make learning fun!
        
        🔍 Search for a topic on Wikipedia.
        📁 Upload a file (.pdf, .txt, or .docx).
        
        Let's get started! Use the sidebar to select your input source.
        """
    )

if not st.session_state["key_test"]:
    st.error("⚠️ Please enter a valid OpenAI API Key to proceed.")
    st.stop()

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = PromptTemplate.from_template(
    """
    You are a helpful assistant that is role playing as a teacher.
                    
    Based ONLY on the following context make 5 questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.

    The difficulty level of the problem is '{level}'.

    Context: {context}
"""
)

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, level):
    chain = prompt | llm
    return chain.invoke({"context": _docs, "level": level})

if not docs:
    st.info("🔑 **API Key Verified!** You are now connected to OpenAI's API. Ready to generate quizzes.")
    st.info("🔍 **Choose Your Source:** Upload a file or search Wikipedia in the sidebar to begin.")
else:
    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[
            StreamingStdOutCallbackHandler(),
        ],
        openai_api_key=key,
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[
            function,
        ],
    )
    response = run_quiz_chain(docs, topic if topic else file.name, level)
    response = response.additional_kwargs["function_call"]["arguments"]
    with st.form("questions_form"):
        questions = json.loads(response)["questions"]
        question_count = len(questions)
        success_count = 0
        for question in questions:
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
                success_count += 1
            elif value is not None:
                st.error("Wrong!")
        if question_count == success_count:
            st.balloons()
        button = st.form_submit_button()
