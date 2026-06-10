# 📚 DocFinder AI

This is my project for searching inside PDF files. You can upload PDFs and then ask questions from them. It will show you the most related text from your documents.


---

## What this project does

- You upload PDF files
- The system reads them and saves them in Pinecone database
- Then you can search anything and it finds the answer from your PDFs
- It uses AI embeddings to understand meaning not just keywords

---

## Tools and Libraries Used

- Python
- Streamlit (for the website interface)
- SentenceTransformers (for converting text to vectors)
- Pinecone (for storing and searching vectors)
- pypdf (for reading PDF files)

---

## How to run

First install the requirements:

```
pip install -r requirements.txt
```

Then run the app:

```
streamlit run app.py
```

---

## How to use

1. Open the app in browser
2. Enter your Pinecone API key in the sidebar (get it free from pinecone.io)
3. Go to Add Documents tab and upload your PDF files
4. Click Start Indexing button
5. Then go to Find Answers tab and type your question
6. It will show matching results with score

---

## Project Files

```
DocFinder-AI/
├── app.py                   
├── requirements.txt         
├── README.md                
├── generate_sample_pdfs.py  
├── sample_pdfs/             
│   ├── 01_Machine_Learning.pdf
│   ├── 02_Internet_of_Things.pdf
│   ├── 03_Cloud_Computing.pdf
│   ├── 04_Natural_Language_Processing.pdf
│   └── 05_Data_Science.pdf
└── utils/
    ├── pdf_processor.py     
    ├── embeddings.py        
    └── pinecone_client.py   
```

---

## Pinecone Setup

- Go to app.pinecone.io and make a free account
- Copy your API key from dashboard
- Paste it in the sidebar when you open the app
- Index will be created automatically

---

## Sample Questions you can try

- What is machine learning and how does it work?
- What are the security risks in IoT devices?
- What are the benefits of cloud computing?
- How does natural language processing understand text?
- What tools do data scientists use?

---

## Author

**Abdul Aziz**
"# ASSIGNMNET-15" 
