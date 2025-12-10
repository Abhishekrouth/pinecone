from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
import uuid,os
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pincone_index_name = os.getenv("PINECONE_INDEX_NAME", "rag-dotproduct")

pc = Pinecone(api_key=pinecone_api_key)

if pincone_index_name not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=pincone_index_name,
        dimension=768,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(pincone_index_name)
print(pc.list_indexes())

def embed_text(text):
    result = genai.embed_content(model="text-embedding-004",content=text)
    return result["embedding"]

def extract_text_from_file(uploaded_file):
    filename = uploaded_file.filename.lower()
    if filename.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
        return text
    return None

def split_text(text, max_length=200):
    words = text.split()
    for i in range(0, len(words), max_length):
        yield " ".join(words[i:i + max_length])

def store_chunks_pinecone(text, source):
    for chunk in split_text(text):
        vector = embed_text(chunk)
        chunk_id = str(uuid.uuid4())
        index.upsert(
            vectors=[(chunk_id, vector, {"text": chunk, "source": source})])

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Corporate Chatbot"

@app.route('/upload_pdf', methods=['POST'])
def upload_docs():
    if "files" not in request.files:
        return jsonify({"error": "Upload your company's PDF file"}), 400    
    uploaded_files = request.files.getlist("files")
    stored_files = []
    for file in uploaded_files:
        text = extract_text_from_file(file)
        store_chunks_pinecone(text, file.filename)
        stored_files.append(file.filename)
    return jsonify({"message": "Files processed", "stored_files": stored_files})

@app.route('/ask_chatbot', methods=['POST'])
def ask_chatbot():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "Enter Question"}), 400

    question_vector = embed_text(question)
    results = index.query(vector=question_vector,top_k=2,include_metadata=True)

    context_text = ""
    for match in results["matches"]:
        context_text += match["metadata"]["text"] + "\n"
    
    prompt = f"""
    You are a Corporate QA assistant. Use only the below context to answer and reply
    "if someone asks a question unrelated to the context of the uploaded document, reply them with "Please ask a relevant question (corporate)." 

    CONTEXT:
    {context_text}

    QUESTION:
    {question}

    ANSWER: """

    response = model.generate_content(prompt)

    if response.text:
        answer = response.text  
    else:
        return f"No answer"

    metadata_list = []
    for match in results['matches']:
        metadata_list.append({
            'id': match['id'],
            'score': match['score'],
            'metadata': match['metadata']
        })

    return jsonify({
        "question": question,
        "answer": answer,
        "source": metadata_list
    })

@app.route('/delete_index', methods = ['POST'])
def del_index():
    pc.delete_index(pincone_index_name)
    return jsonify({"message": "Index deleted"})

if __name__ == '__main__':
    app.run(debug=True)