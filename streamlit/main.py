from .DataBaseChatbot import DataBaseChat
print_results=False  #Prints results(query,data,summary,plot) immediatly after they are generated
auto_train=False     #Keep False to avoid adding wrong queries to vector database
result_summary=True #Generates text summary for output data 
visualize=False      #generates plot for the outpput data

def run_code(question,table_key,username,mode):
    user=username
    table_key_dict = {
        'congestion': 'public.congestion',
        'toll_plaza_data': 'public.toll_plaza_data',
    }

    #user editable depending on query db
    if table_key== "congestion" or "toll_plaza_data":
        table_name=table_key_dict[table_key]
    else:
        print("\nENTER VALID TABLE_KEY\n")
        raise ValueError("An error occurred: invalid table_key entered")

    your_instance = DataBaseChat(table_key,table_name)
    your_instance.main_setup()
    print("question:: ",question)
    if mode=='ASK':
        sql,df,text_summary,plot = your_instance.ask(question=question,print_results = print_results,auto_train = auto_train,result_summary = result_summary,visualize = visualize,username=user)
        return sql,df,text_summary,plot
    # your_instance.ask_function_in_loop(tables,print_results = print_results,auto_train = auto_train,result_summary = result_summary,visualize = visualize)
