from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from langchain_openai.embeddings import OpenAIEmbeddings
import tkinter as tk

paragraph = ""
API= #ENTER YOUR API KEY
path_folder= #PLS ENTER YOUR FOLDER PATH
conts = os.listdir(path_folder)


#savelocal=input("give any name to save files in batch")
for file_name in conts:
    path=(path_folder+"/"+file_name)
    print(path)
    #path = os.path.join(path_folder, file_name)
    pdreader = PdfReader(path)
    for i, page in enumerate(pdreader.pages):
        content = page.extract_text()
        if content:
            paragraph += content
print(paragraph)
# Split the text
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
texts = text_splitter.split_text(paragraph)
# Generate embeddings
embeddings = OpenAIEmbeddings(openai_api_key=API)
#----
# Process texts in smaller batches to avoid hitting the rate limit
def faiss_saver():
    try:
        if os.path.isdir("unicadbfold")==False:
            maxiterate =0
        else:
            maxiterate =os.listdir("unicadbfold")
            maxiterate = max([int(item.split('_')[1]) for item in maxiterate])+1
        print(len(texts))
        print(maxiterate)
        batch_size = 100  # Adjust this value based on your rate limit
        if maxiterate != (len(texts) // batch_size):
            for i in range(maxiterate*batch_size, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                storage = FAISS.from_texts(batch_texts, embeddings)
                # Save the embeddings locally
                storage.save_local(f"unicadbfold/Unicadb_{i//batch_size}")
                print(i//batch_size)

        new_db = None
        templent = os.listdir("unicadbfold")
        print(templent)
        for i in templent:
            tam = ("unicadbfold/{}").format(i)
            if new_db is None:
                new_db = FAISS.load_local(tam, embeddings)
                print(i)
            else:
                batch_storage = FAISS.load_local(tam, embeddings)
                new_db.merge_from(batch_storage)
                print(i)
        new_db.save_local("uncaDBall")
    except:
        faiss_saver()

faiss_saver()
#-----


from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

chain = load_qa_chain(OpenAI(openai_api_key=API), chain_type="stuff")

embeddings = OpenAIEmbeddings(openai_api_key=API)
def search():

    query = input("enter")
    storage = FAISS.load_local("uncaDBall", embeddings)

    docs = storage.similarity_search(query)
    tas = chain.run(input_documents=docs, question=query)
    print(tas)


root = tk.Tk()
root.title("poc")


root.configure(bg='#f4fec9')

company_logo = tk.PhotoImage(file='Tamil.png')

logo_label = tk.Label(root, image=company_logo, bg='#f4fec9')
logo_label.pack(anchor='nw', padx=20, pady=20)

frame = tk.Frame(root, bg='#f4fec9')
frame.pack(padx=40, pady=80)

entry = tk.Entry(frame, width=80)
entry.pack(side=tk.LEFT, padx=10)

search_button = tk.Button(frame, text="Search", command=search)
search_button.pack(side=tk.LEFT, padx=10)

result_label = tk.Label(root, text="Search Results will appear here", font=('Helvetica', 12), wraplength=300,
                        bg='#f4fec9')
result_label.pack(padx=40, pady=20)

root.mainloop()


