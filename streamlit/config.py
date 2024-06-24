class config():
    #Database
    HOST='localhost'
    DB_NAME='congestion'
    DB_USER='postgres'
    DB_PASSWORD='Godlygodzilla'
    DB_PORT='5432'

    #format [schema.table_name]
    tables_in_db=['public.congestion','public.toll_plaza_data']
    #questions and queries
    path_congestion=r"C:\Users\vinee\Downloads\IBI\codes\text2sql\vaana\questions_ans_database_congestion.npy"
    path_trafficflow=r"C:\Users\vinee\Downloads\IBI\codes\text2sql\vaana\questions_ans_database_trafficflow.npy"
    #Ollama
    ollama_host= "https://solid-toad-mature.ngrok-free.app"
    code_model= "codestral:22b" #"codellama:7b" "codellama:13b"    #choose between "codellama:7b" or "codellama:13b" or "codestral:22b"
    text_model= "llama2:7b" #Only llama2:7b installed currently 
    #Chroma DB
    number_of_questions_for_RAG=5
    chroma_client_host="13.127.163.60"
    chroma_client_port="8000"
