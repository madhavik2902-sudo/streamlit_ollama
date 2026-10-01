import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* ==========================================
   MAIN BACKGROUND
   ========================================== */

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(79, 70, 229, 0.35),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 90%,
            rgba(147, 51, 234, 0.30),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #020617,
            #0f172a,
            #172554,
            #1e1b4b
        );

    color: white;
    font-family: "Segoe UI", sans-serif;
}


/* ==========================================
   MAIN CONTAINER
   ========================================== */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}


/* ==========================================
   TITLE
   ========================================== */

h1 {
    text-align: center;
    color: #ffffff !important;

    font-size: 48px !important;

    font-weight: 800;

    margin-bottom: 8px;

    text-shadow:
        0px 0px 10px rgba(129, 140, 248, 0.8),
        0px 0px 25px rgba(139, 92, 246, 0.5);
}


/* ==========================================
   SUBTITLE
   ========================================== */

.stApp p {
    color: #dbeafe;
}


/* ==========================================
   HEADINGS
   ========================================== */

h2 {
    color: #c4b5fd !important;
}

h3 {
    color: #a5b4fc !important;
}


/* ==========================================
   INFO MESSAGE
   ========================================== */

[data-testid="stAlert"] {

    background: rgba(30, 41, 59, 0.80);

    border-radius: 16px;

    border: 1px solid rgba(129, 140, 248, 0.45);

    box-shadow:
        0 8px 25px rgba(0, 0, 0, 0.25);
}


/* ==========================================
   PDF UPLOADER
   ========================================== */

[data-testid="stFileUploader"] {

    background:
        linear-gradient(
            135deg,
            rgba(30, 41, 59, 0.95),
            rgba(49, 46, 129, 0.75)
        );

    padding: 25px;

    border-radius: 22px;

    border: 2px dashed #818cf8;

    box-shadow:
        0 10px 35px rgba(0, 0, 0, 0.35);

    margin-top: 20px;

    transition: 0.3s;
}


/* ==========================================
   PDF UPLOADER HOVER
   ========================================== */

[data-testid="stFileUploader"]:hover {

    border-color: #c084fc;

    box-shadow:
        0 12px 40px rgba(139, 92, 246, 0.30);
}


/* ==========================================
   FILE UPLOADER TEXT
   ========================================== */

[data-testid="stFileUploader"] label {

    color: #ffffff !important;

    font-weight: 600;
}


/* ==========================================
   TEXT INPUT
   ========================================== */

[data-testid="stTextInput"] input {

    background: rgba(15, 23, 42, 0.95) !important;

    color: white !important;

    border: 2px solid #6366f1 !important;

    border-radius: 15px !important;

    padding: 14px !important;

    font-size: 16px !important;

    box-shadow:
        0 5px 20px rgba(0, 0, 0, 0.20);

    transition: 0.3s;
}


/* ==========================================
   INPUT FOCUS
   ========================================== */

[data-testid="stTextInput"] input:focus {

    border-color: #c084fc !important;

    box-shadow:
        0 0 15px rgba(168, 85, 247, 0.45) !important;
}


/* ==========================================
   BUTTONS
   ========================================== */

.stButton > button {

    width: 100%;

    background:
        linear-gradient(
            90deg,
            #4f46e5,
            #7c3aed,
            #9333ea
        );

    color: white;

    border: none;

    border-radius: 14px;

    padding: 13px 20px;

    font-size: 16px;

    font-weight: 700;

    box-shadow:
        0 8px 20px rgba(124, 58, 237, 0.35);

    transition: all 0.3s ease;
}


/* ==========================================
   BUTTON HOVER
   ========================================== */

.stButton > button:hover {

    transform: translateY(-3px);

    box-shadow:
        0 12px 30px rgba(139, 92, 246, 0.55);

    color: white;

    border: none;
}


/* ==========================================
   EXPANDER
   ========================================== */

[data-testid="stExpander"] {

    background:
        rgba(15, 23, 42, 0.90);

    border:

        1px solid

        rgba(129, 140, 248, 0.50);

    border-radius: 18px;

    box-shadow:
        0 8px 25px rgba(0, 0, 0, 0.30);

    margin-top: 15px;
}


/* ==========================================
   EXPANDER TEXT
   ========================================== */

[data-testid="stExpander"] p {

    color: #e2e8f0 !important;

    line-height: 1.7;
}


/* ==========================================
   SUCCESS MESSAGE
   ========================================== */

[data-testid="stAlert"] p {

    color: #f8fafc !important;
}


/* ==========================================
   DIVIDER
   ========================================== */

hr {

    border: none;

    height: 2px;

    background:
        linear-gradient(
            90deg,
            transparent,
            #6366f1,
            #a855f7,
            transparent
        );

    margin: 35px 0;
}


/* ==========================================
   SPINNER
   ========================================== */

[data-testid="stSpinner"] {

    color: #a78bfa;
}


/* ==========================================
   REMOVE STREAMLIT MENU
   ========================================== */

#MainMenu {

    visibility: hidden;
}


/* ==========================================
   REMOVE FOOTER
   ========================================== */

footer {

    visibility: hidden;
}


/* ==========================================
   FILE NAME CARD
   ========================================== */

[data-testid="stFileUploaderFile"] {

    background: rgba(30, 41, 59, 0.95);

    border-radius: 12px;

    border: 1px solid #6366f1;
}


/* ==========================================
   METRIC / INFORMATION TEXT
   ========================================== */

[data-testid="stMetricValue"] {

    color: #c4b5fd !important;
}


/* ==========================================
   ANSWER AREA
   ========================================== */

.stMarkdown {

    line-height: 1.7;
}


/* ==========================================
   SCROLLBAR
   ========================================== */

::-webkit-scrollbar {

    width: 8px;
}

::-webkit-scrollbar-track {

    background: #0f172a;
}

::-webkit-scrollbar-thumb {

    background: linear-gradient(
        #6366f1,
        #a855f7
    );

    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# TITLE
# ==========================================

st.title("📄 Chat with Your PDF")

st.write(
    "🤖 Ask questions about your PDF using RAG + Ollama"
)


# ==========================================
# TEST MESSAGE
# ==========================================

st.info(
    "✨ Application is running successfully!"
)


# ==========================================
# PDF UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "📂 Upload your PDF",
    type=["pdf"]
)


# ==========================================
# PROCESS PDF
# ==========================================

if uploaded_file is not None:

    st.success(
        "✅ PDF uploaded successfully!"
    )


    # Read PDF

    reader = PdfReader(
        uploaded_file
    )


    st.write(
        "📑 Number of pages:",
        len(reader.pages)
    )


    # Extract text

    text = ""


    for page in reader.pages:

        page_text = page.extract_text()


        if page_text:

            text += page_text + "\n"


    # Check text

    if not text.strip():

        st.error(
            "❌ No text could be extracted from this PDF."
        )

        st.stop()


    st.write(
        "📝 Total characters:",
        len(text)
    )


    # ==========================================
    # CREATE CHUNKS
    # ==========================================

    chunk_size = 500

    chunks = []


    for i in range(
        0,
        len(text),
        chunk_size
    ):

        chunk = text[
            i:i + chunk_size
        ]


        if chunk.strip():

            chunks.append(
                chunk
            )


    st.write(
        "🧩 Number of chunks:",
        len(chunks)
    )


    # ==========================================
    # LOAD EMBEDDING MODEL
    # ==========================================

    with st.spinner(
        "🧠 Loading embedding model..."
    ):

        embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )


    st.success(
        "✅ Embedding model loaded!"
    )


    # ==========================================
    # CREATE EMBEDDINGS
    # ==========================================

    with st.spinner(
        "🔄 Creating embeddings..."
    ):

        embeddings = embedding_model.encode(
            chunks
        )


    st.success(
        "✅ Embeddings created!"
    )


    # ==========================================
    # CREATE CHROMADB
    # ==========================================

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )


    collection = client.get_or_create_collection(
        name="documents"
    )


    # ==========================================
    # STORE DOCUMENT
    # ==========================================

    ids = []


    for i in range(
        len(chunks)
    ):

        ids.append(
            "chunk_" + str(i)
        )


    # Remove old data

    try:

        old_data = collection.get()


        if old_data["ids"]:

            collection.delete(
                ids=old_data["ids"]
            )


    except Exception:

        pass


    collection.add(

        ids=ids,

        documents=chunks,

        embeddings=embeddings.tolist()

    )


    st.success(
        "💾 PDF stored successfully in ChromaDB!"
    )


    st.divider()


    # ==========================================
    # QUESTION
    # ==========================================

    question = st.text_input(
        "💬 Ask a question about the PDF"
    )


    # ==========================================
    # ANSWER
    # ==========================================

    if question:

        with st.spinner(
            "🔎 Searching document..."
        ):

            # Create question embedding

            question_embedding = (
                embedding_model.encode(
                    [question]
                )
            )


            # Search ChromaDB

            results = collection.query(

                query_embeddings=
                question_embedding.tolist(),

                n_results=len(chunks)

            )


            # Get documents

            relevant_chunks = (
                results["documents"][0]
            )


            # Combine context

            context = "\n\n".join(
                relevant_chunks
            )


        # ==========================================
        # SHOW RETRIEVED DATA
        # ==========================================

        with st.expander(
            "📖 View Retrieved Context"
        ):

            st.write(
                context
            )


        # ==========================================
        # CREATE PROMPT
        # ==========================================

        prompt = f"""

You are a helpful document assistant.

Answer the user's question using the information
available in the document context.

If the user asks for a summary, overview, or brief
description, summarize the important information
from the document context.

If the answer cannot be found in the document,
say:

"I could not find the answer in the document."

Do not make up information.

Document Context:

{context}

User Question:

{question}

Answer:

"""


        # ==========================================
        # OLLAMA
        # ==========================================

        with st.spinner(
            "🤖 Ollama is generating the answer..."
        ):

            response = ollama.chat(

                model="llama3.2",

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

            )


        # ==========================================
        # DISPLAY ANSWER
        # ==========================================

        answer = response[
            "message"
        ][
            "content"
        ]


        st.subheader(
            "🤖 Answer"
        )


        st.write(
            answer
        )