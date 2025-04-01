# Importing Required Libraries
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.retrievers.multi_query import MultiQueryRetriever
from concurrent.futures import ThreadPoolExecutor


# -----------------------------------------------------------------------------------------
# 🔹 Environment Setup & Defining AI Model
# -----------------------------------------------------------------------------------------

# Set API key for Google Gemini model
os.environ["GOOGLE_API_KEY"] = "AIzaSyBgXK36EMVy1m8mQAEDpo4feTdjUD2hZi0"

# Initialize Gemini language model
llm_gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4)



# -----------------------------------------------------------------------------------------
# 🔹 Defining Embedding Model
# -----------------------------------------------------------------------------------------

# Load sentence-transformers for embedding text into vector space
hf_embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={"device": "cpu"},  # Running on CPU
    encode_kwargs={"normalize_embeddings": True}
)



# -----------------------------------------------------------------------------------------
# 🔹 Loading FAISS Vectorstore
# -----------------------------------------------------------------------------------------

# Path to pre-built FAISS database
DB_FAISS_PATH = "Vectorstore/faiss_db"

# Load FAISS vector store containing indexed medical documents
vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings=hf_embeddings, allow_dangerous_deserialization=True)



# -----------------------------------------------------------------------------------------
# 🔹 Reranking Retrieved Documents
# -----------------------------------------------------------------------------------------

# Load cross-encoder model for reranking search results
cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")

def rerank_documents(docs, query):
    """Reranks retrieved documents using a cross-encoder model based on query relevance."""
    pairs = [(query, doc.page_content) for doc in docs]
    with ThreadPoolExecutor() as executor:
        scores = list(executor.map(lambda pair: cross_encoder.predict([pair])[0], pairs))
    sorted_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc[0] for doc in sorted_docs][:15]  # Return top 15 most relevant documents



# -----------------------------------------------------------------------------------------
# 🔹 MultiQuery Retriever for Diverse Query Representation
# -----------------------------------------------------------------------------------------

# Enables retrieval of diverse document sets by rephrasing queries
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm_gemini,
    include_original=True  
)

def retrieve_and_rerank(query):
    """Retrieves relevant documents and applies reranking."""
    retrieved_docs = multiquery_retriever.invoke(query)
    return rerank_documents(retrieved_docs, query)



# -----------------------------------------------------------------------------------------
# 🔹 Prompt for Medical Chatbot
# -----------------------------------------------------------------------------------------

# Structured prompt to generate a detailed and well-organized medical response
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an **AI-powered medical assistant** trained to provide accurate, reliable, and actionable medical information.
    Your responses must be: **precise, evidence-based, and safety-conscious**.
    Your reponse must be organised and to the point.

    **Response Guidelines:**
    1. **Relevance**: Only address the specific medical query asked - don't volunteer extra information
    2. **Structure**: Use ONLY the relevant sections below based on the question
    3. **Priority**: Always highlight urgent medical concerns first
    4. **Clarity**: Use simple language but maintain medical accuracy
    5. **Safety**: Include disclaimers for serious conditions

    **Available Response Sections (Use only as needed):**

    ### Condition Overview:
    - When asked "what is..." or for general explanations
    - Brief definition (1 sentence)
    - Key characteristics (2-3 bullet points)

    ### Symptoms:
    - When asked about signs or symptoms
    - List by frequency (Common → Rare)
    - ★ Mark dangerous symptoms requiring immediate care

    ### Causes & Risk Factors:
    - When asked "why" or "how do you get..."
    - Primary causes
    - Modifiable vs non-modifiable risk factors

    ### Diagnosis:
    - When asked about tests or identification
    - Common diagnostic methods
    - Differential diagnosis considerations

    ### Treatment Options:
    - Medical therapies (gold standard first)
    - Lifestyle modifications
    - Emerging treatments (if relevant)

    ### When to Seek Care
    - ALWAYS include if symptoms are mentioned
    - Specific warning signs
    - Recommended specialist (if applicable)

    ### Prevention
    - Evidence-based prevention strategies
    - Screening recommendations if appropriate

    **Safety Disclaimers (include when appropriate):**
    - "This information cannot replace professional medical advice..."
    - "If you're experiencing [X serious symptom], seek emergency care immediately"
    - "Consult your physician before starting any treatment"

    **User's Question:** {question}

    **Medical Context:** {context}

    **Your Response:**
    """
)

# -----------------------------------------------------------------------------------------
# 🔹 RAG Pipeline (Retrieval-Augmented Generation)
# -----------------------------------------------------------------------------------------

def format_docs(docs):
    """Formats retrieved documents into a single string."""
    return '\n\n'.join([doc.page_content for doc in docs])

# Define RAG pipeline using document retrieval and generative AI model
rag_chain = (
    {"context": RunnableLambda(retrieve_and_rerank) | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm_gemini
    | StrOutputParser()
)



# -----------------------------------------------------------------------------------------
# 🔹 Evaluation Function for AI-Generated Responses
# -----------------------------------------------------------------------------------------

def evaluate_response(response):
    """Evaluates response quality based on key medical criteria using Gemini."""
    evaluation_prompt = PromptTemplate(
        input_variables=["response"],
        template="""
        You are a **self-critical AI medical assistant**. Your task is to evaluate the quality of the following response.

        **Response:**
        {response}

        **Evaluation Instructions:**
        Rate the response on the following **medical-specific criteria** (1-10):
        1. **Medical Accuracy**: Alignment with established medical guidelines.
        2. **Clinical Relevance**: Direct relevance to the user's query.
        3. **Actionability**: Provides clear, actionable steps (e.g., treatment, prevention, when to see a doctor).
        4. **Comprehensiveness**: Covers all critical aspects of the query (e.g., symptoms, causes, treatments, prevention).

        **Evaluation Format:**
        - Medical Accuracy: [score]
        - Clinical Relevance: [score]
        - Actionability: [score]
        - Comprehensiveness: [score]

        **Evaluation:**
        """
    )
    evaluation_chain = evaluation_prompt | llm_gemini | StrOutputParser()
    evaluation = evaluation_chain.invoke({"response": response})
    return evaluation



# -----------------------------------------------------------------------------------------
# 🔹 Iterative Refinement Chatbot with Query Classification
# -----------------------------------------------------------------------------------------

def extract_scores(evaluation):
    """Extracts numerical evaluation scores from AI-generated text."""
    try:
        return tuple(int(evaluation.split(f"{category}: ")[1].split("\n")[0])
                     for category in ["Medical Accuracy", "Clinical Relevance", "Actionability", "Comprehensiveness"])
    except (IndexError, ValueError):
        return 0, 0, 0, 0

def classify_query(query):
    """Classifies the user query as either 'medical' or 'non-medical' using Gemini."""
    classification_prompt = f"""
    Classify the following query as either "medical" or "non-medical":
    Query: {query}
    Classification:
    """
    return llm_gemini.invoke(classification_prompt).content.strip().lower()

def chatbot_with_iterative_refinement(query, max_iterations=3):
    """Handles user queries, refining responses iteratively for medical queries, and routing non-medical queries to Gemini."""
    
    classification = classify_query(query)

    if classification == "medical":
        response = rag_chain.invoke(query)
        
        for iteration in range(max_iterations):
            evaluation = evaluate_response(response)
            scores = extract_scores(evaluation)

            # Stop if at least 3 out of 4 criteria meet the quality threshold
            if sum(score >= 8 for score in scores) >= 3:
                break

            response = rag_chain.invoke(query)  # Re-run retrieval and generation
    
        return response

    else:
        # Respond to non-medical queries
        response = llm_gemini.invoke(f"""
        Remember this while ansering any user query that you are a medical chatbot designed to provide accurate and concise medical advice. 
        Do not mention anywhere you are a gemini build by google or anything like that.
        Answer as a medical bot which is here to help user related query and for other queries just you i am a medical chatbot and cannot help you with any non medical related query.

        User Query: {query}
        Response:
        """).content
        return response



# -----------------------------------------------------------------------------------------
# 🔹 Example Execution
# -----------------------------------------------------------------------------------------

if __name__ == "__main__":
    query = "my mother is having leg swelling from quiet a time and after full body checkup, she only got thyroid issue so, why her legs are swelling and how to treat them"
    final_answer = chatbot_with_iterative_refinement(query)
    print("\nFinal Answer:", final_answer)
