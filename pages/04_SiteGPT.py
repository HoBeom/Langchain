from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
import os
import re

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
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
                    max_tokens=10,
                )
                llm_test.predict("Test connection.")
                st.success("✅ API Key is valid!")
                st.session_state["key_test"] = True
            except Exception:
                st.error(f"❌ Invalid API Key")
                st.session_state["key_test"] = False
        else:
            st.success("✅ API Key is valid!")
    model_select = st.selectbox("Model", ("gpt-4o-mini", "gpt-3.5-turbo-1106"))
    st.markdown("---")
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )
    st.markdown("---")
    st.markdown("Github: [hobeom/langchain](https://github.com/hobeom/langchain)")

# llm = ChatOpenAI(
#     temperature=0.1,
# )

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context, answer the user’s question. You must analyze and extract all relevant information from the context to answer as accurately as possible. If the information can be logically inferred based on the provided details, you should do so. However, if there is insufficient data to answer the question, respond with “I don’t know” and nothing else.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
    ---                                              
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
    ---                                              
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )

def parse_page(soup):
    # target_classes = ["feedback", "navigation", "header", "footer", "breadcrumb", 
    #               "pagination-links", "cookie-settings", "meta", "footer-links", 
    #               "social-media", "prompt", "DocsFooter"]
    # target_elements = soup.find_all(class_=target_classes)
    # # print(f"Found {len(target_elements)} elements to decompose.")
    # for element in target_elements:
    #     element.decompose()
    # markdown = soup.find_all("div", class_=re.compile(".*markdown.*"))
    # # print(f"Found {len(markdown)} markdown elements.")
    # if markdown: # TODO: Test document infomation loss
    #     markdown_text = " ".join(div.get_text() for div in markdown).split()
    #     return " ".join(markdown_text)
    main_content = soup.find("div", class_="main-pane")
    if not main_content:
        main_content = soup.find(['main', 'article']) or soup.find(attrs={"role": "main"})
        if not main_content:
            main_content = soup.find('div', class_=['content', 'page-content', 'post-entry', 'text'])
    if main_content:
        return ' '.join(main_content.get_text().split())
    return ' '.join(soup.get_text().split())

@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
        filter_urls=[
            r"^(.*\/ai-gateway\/.*).*",
            r"^(.*\/workers-ai\/.*).*",
            r"^(.*\/vectorize\/.*).*",
            r".*billing.*",
        ]
    )
    # workers-ai
    # ai-gateway
    # vectorize
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    parse_url = re.sub(r"[^a-zA-Z0-9]", "_", url)
    embed_dir = f"./.cache/sitegpt/{parse_url}"
    os.makedirs(embed_dir, exist_ok=True)
    cache_dir = LocalFileStore(embed_dir)
    embeddings = OpenAIEmbeddings(openai_api_key=key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vector_store = FAISS.from_documents(docs, cached_embeddings)
    return vector_store.as_retriever()

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:

        llm = ChatOpenAI(
            temperature=0.1,
            openai_api_key=key,
            model=model_select,
            # streaming=True, # TODO chat mode
        )
        retriever = load_website(url)
        query = st.text_input("Ask a question to the website.")
        if query:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            result = chain.invoke(query) 
            # TODO: memoize the result
            st.markdown(result.content.replace("$", "\$"))
