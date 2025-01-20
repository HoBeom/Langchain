import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
import regex as re
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser
from langchain.schema import HumanMessage

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        result = None
        text = text.replace("```", "").replace("json", "")
        try:
            result = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error: {e}")
        if result is not None:
            return result
        try:
            json_pattern = r"\{(?:[^{}]|(?R))*\}"
            match = re.search(json_pattern, text, re.DOTALL)
            if not match:
                raise ValueError("No valid JSON found in the input text.")
            json_str = match.group(0)
            parsed_json = json.loads(json_str)
            return parsed_json
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error: {e}")


output_parser = JsonOutputParser()

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

llm = ChatOpenAI(
    temperature=0.1,
    model=model_select,
    openai_api_key=key,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 5 (FIVE) questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    The difficulty level of the problem is '{level}'.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!
         
    Context: {context}
""",
        )
    ]
)

# questions_chain = {"context": format_docs } | questions_prompt | llm
questions_chain = questions_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }}
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }}
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }}
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }}
                ]
            }}
        ]
     }}
    ```
    Your turn!

    Questions: {context}

""",
        )
    ]
)

formatting_chain = formatting_prompt | llm

@st.cache_data(show_spinner="Making quiz...")
def generate_questions(_docs, level):
    return questions_chain.invoke({"context": format_docs(_docs), "level": level}).content

@st.cache_data(show_spinner="Formatting quiz...")
def format_questions(qize):
    chain = formatting_chain | output_parser
    return chain.invoke({"context": qize})

if not docs:
    st.info("🔑 **API Key Verified!** You are now connected to OpenAI's API. Ready to generate quizzes.")
    st.info("🔍 **Choose Your Source:** Upload a file or search Wikipedia in the sidebar to begin.")
else:
    response = format_questions(generate_questions(docs, level=level))
    with st.form("questions_form"):
        # st.write(response)
        questions = response["questions"]
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
