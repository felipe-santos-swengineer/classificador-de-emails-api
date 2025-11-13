# Email Classifier API (Backend)

API para classificar e-mails como **Produtivo** ou **Improdutivo** e gerar respostas automáticas usando **Hugging Face Llama 3.1**.

---

## ⚡ Funcionalidades

- Classificação automática de e-mails: **Produtivo / Improdutivo**  
- Geração de resposta contextual para e-mails produtivos e educada para improdutivos  
- Aceita **texto colado** ou arquivos **.txt / .pdf**  
- Backend em **FastAPI** com integração à Hugging Face Inference API  

---

## 🛠️ Tecnologias

- Python 3.10+  
- FastAPI  
- Hugging Face Inference API (`meta-llama/Llama-3.1-8B-Instruct`)  
- pdfplumber (para PDFs)  
- nltk (para pré-processamento de texto)  

---

## 🚀 Instalação

1. Clone o repositório:

2. criar virtualenv e ativar
   python -m venv venv
   source venv/bin/activate   # linux/mac
   venv\Scripts\activate      # windows

3. Crie um .env e adicione:
   HUGGINGFACEHUB_API_TOKEN=seu_token_aqui

4. instalar dependências
   pip install -r requirements.txt

5. uvicorn main:app --reload

A aplicação estará disponível em http://localhost:8000

