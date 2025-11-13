import os
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import re
import pdfplumber
import nltk
import io

# pré processamento NLP  #
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Config da API e do HF #
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    raise ValueError("⚠️ Token Hugging Face não encontrado no .env")

app = FastAPI(
    title="Email Classifier API",
    version="4.0.0",
    description="Classifica e gera respostas automáticas para e-mails (Produtivo ou Improdutivo)."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = InferenceClient(token=HF_TOKEN)
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def tokenize_text(text: str):
    return re.findall(r'\b\w+\b', text.lower())

def extract_text_from_pdf(file: bytes) -> str:
    with pdfplumber.open(io.BytesIO(file)) as pdf:
        return " ".join(page.extract_text() or "" for page in pdf.pages)

def extract_text_from_txt(file: bytes) -> str:
    return file.decode("utf-8")

def preprocess_text(text: str) -> str:
    tokens = tokenize_text(text) #nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words("portuguese"))
    lemmatizer = WordNetLemmatizer()
    filtered = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(filtered)

# Prompt para classificar o email #
def classify_email(text: str) -> str:
    """Classifica e-mail como Produtivo ou Improdutivo."""
    prompt = f"""
Você é um assistente profissional que separa emails que precisam ou não de retorno.  

Definições:
- 'Produtivo' -> Emails que requerem uma ação ou resposta específica (ex.: solicitações de suporte técnico, atualização sobre casos em aberto, dúvidas sobre o sistema).
- 'Improdutivo' -> Emails que não necessitam de uma ação imediata e retorno (ex.: mensagens de felicitações, agradecimentos).

Exemplos:
- 'Não consigo acessar minha conta, preciso de ajuda.' -> Produtivo
- 'Obrigado pelo suporte ontem, foi ótimo.' -> Improdutivo
- 'Preciso atualizar meu cadastro no sistema.' -> Produtivo
- 'Parabéns pelo excelente atendimento.' -> Improdutivo

Classifique o e-mail abaixo como Produtivo ou Improdutivo.

E-mail:
{text}

Responda apenas com: Produtivo ou Improdutivo.
"""

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.0,
    )
    output = completion.choices[0].message["content"].strip().lower()

    if output == "produtivo":
        return "Produtivo"
    elif output == "improdutivo":
        return "Improdutivo"
    else:
        print("⚠️ Classificação inesperada:", output)
        return "Produtivo"

# Prompt para responder o email, de acordo com categoria e duvida #
def generate_response(category: str, text: str) -> str:
    """Gera resposta considerando o conteúdo real do e-mail."""
    if category == "Produtivo":
        prompt = f"""
Você é um assistente profissional.  
Responda ao seguinte e-mail de forma positiva, cordial, agradecendo o contato, 
confirmando o recebimento e abordando a dúvida do usuário:

E-mail: {text}
"""
    else:
        prompt = f"""
Você é um assistente profissional.  
Responda ao seguinte e-mail de forma educada, cordial e breve, 
informando que não há necessidade de retorno:

E-mail: {text}
"""
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.7,
    )
    return completion.choices[0].message["content"].strip()

# API #
@app.post("/classify")
async def process_email(
    text: str = Form(None),
    file: UploadFile = File(None),
):
    if not text and not file:
        return {"erro": "Nenhum texto ou arquivo enviado."}
    
    """Processa texto ou arquivo e retorna classificação + resposta."""
    if file:
        content = await file.read()
        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(content)
        elif file.filename.endswith(".txt"):
            text = extract_text_from_txt(content)
        else:
            return {"erro": "Formato de arquivo não suportado."}
    elif not text:
        return {"erro": "Nenhum texto ou arquivo enviado."}

    cleaned = preprocess_text(text)
    categoria = classify_email(cleaned)
    resposta = generate_response(categoria, text)

    return {
        "categoria": categoria,
        "resposta": resposta,
        "texto_processado": cleaned[:300] + "..." if len(cleaned) > 300 else cleaned
    }

@app.get("/")
def root():
    return {"mensagem": "API está funcionando!"}
