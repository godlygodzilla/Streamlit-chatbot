from langchain_community.llms import Ollama
import re
import traceback
from typing import List, Tuple, Union
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from dataclasses import dataclass
import json
import dataclasses
import chromadb
from chromadb.utils import embedding_functions
import psycopg2
import numpy as np
import hashlib
import uuid
import time
from PIL import Image
from io import BytesIO
from .config import config
from datetime import datetime

@dataclass
class TrainingPlanItem:
    item_type: str
    item_group: str
    item_name: str
    item_value: str
    database: str

    def __str__(self):
        if self.item_type == self.ITEM_TYPE_SQL:
            return f"Train on SQL: {self.item_group} {self.item_name}"
        elif self.item_type == self.ITEM_TYPE_DDL:
            return f"Train on DDL: {self.item_group} {self.item_name}"
        elif self.item_type == self.ITEM_TYPE_IS:
            return f"Train on Information Schema: {self.item_group} {self.item_name}"

    ITEM_TYPE_SQL = "sql"
    ITEM_TYPE_DDL = "ddl"
    ITEM_TYPE_IS = "is"

class ValidationError(BaseException):
    """Raise for validations"""

    pass


class TrainingPlan:
    
    _plan: List[TrainingPlanItem]

    def __init__(self, plan: List[TrainingPlanItem]):
        self._plan = plan

    def __str__(self):
        return "\n".join(self.get_summary())

    def __repr__(self):
        return self.__str__()

    def get_summary(self) -> List[str]:

        return [f"{item}" for item in self._plan]

    def remove_item(self, item: str):
       
        for plan_item in self._plan:
            if str(plan_item) == item:
                self._plan.remove(plan_item)
                break

@dataclass
class StringData:
    data: str

@dataclass
class StatusWithId:
    success: bool
    message: str
    id: str


class DataBaseChat():
        
    def __init__(self,table_key,table_name):
        self.related_training_data = {}  # Initialize it as an empty list
        self.run_sql_is_set = False
        self.static_documentation = ""
        self.conn = False
        self.related_training_data = {}
        self.ollama_host=config.ollama_host
        self.db_being_used=table_key
        self.table_name=table_name

        default_ef = embedding_functions.DefaultEmbeddingFunction()
        self.embedding_function = default_ef
        self.n_results = config.number_of_questions_for_RAG
        try:
            self.chroma_client=chromadb.HttpClient(host=config.chroma_client_host,port=config.chroma_client_port)
        except:
            raise ValueError(f"Error connecting to chromadb: {self.chroma_client}")

        self.documentation_collection = self.chroma_client.get_or_create_collection(
            name="documentation", embedding_function=self.embedding_function
        )
        self.ddl_collection = self.chroma_client.get_or_create_collection(
            name="ddl", embedding_function=self.embedding_function
        )
        self.sql_collection = self.chroma_client.get_or_create_collection(
            name="sql", embedding_function=self.embedding_function
        )

        self.user_questions_storage_collection = self.chroma_client.get_or_create_collection(
            name="user_questions", embedding_function=self.embedding_function
        )
        
    def connect_to_sql(self,DB_NAME,DB_USER,DB_PASSWORD,DB_PORT,HOST):
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            host=HOST
        )
        if conn:
            self.run_sql_is_set=True
            self.conn=conn
        return conn

    def run_initial_sql(self,sql :str,DB_NAME,DB_USER,DB_PASSWORD,DB_PORT,HOST):
        
        conn=self.connect_to_sql(DB_NAME,DB_USER,DB_PASSWORD,DB_PORT,HOST)
        
        if conn:
            try:
                cs = conn.cursor()
                cs.execute(sql)
                results = cs.fetchall()

                # Create a pandas dataframe from the results
                df = pd.DataFrame(
                    results, columns=[desc[0] for desc in cs.description]
                )
                return df

            except psycopg2.Error as e:
                conn.rollback()
                raise ValidationError(e)

            except Exception as e:
                conn.rollback()
                raise e
    

    def get_training_plan_generic(self, df) -> TrainingPlan:

        # For each of the following, we look at the df columns to see if there's a match:
        database_column = df.columns[
            df.columns.str.lower().str.contains("database")
            | df.columns.str.lower().str.contains("table_catalog")
        ].to_list()[0]
        schema_column = df.columns[
            df.columns.str.lower().str.contains("table_schema")
        ].to_list()[0]
        table_column = df.columns[
            df.columns.str.lower().str.contains("table_name")
        ].to_list()[0]
        columns = [database_column,
                    schema_column,
                    table_column]
        candidates = ["column_name",
                      "data_type",
                      "comment"]
        matches = df.columns.str.lower().str.contains("|".join(candidates), regex=True)
        columns += df.columns[matches].to_list()

        plan = TrainingPlan([])

        for database in df[database_column].unique().tolist():
            for schema in (
                df.query(f'{database_column} == "{database}"')[schema_column]
                .unique()
                .tolist()
            ):
                for table in (
                    df.query(
                        f'{database_column} == "{database}" and {schema_column} == "{schema}"'
                    )[table_column]
                    .unique()
                    .tolist()
                ):
                    df_columns_filtered_to_table = df.query(
                        f'{database_column} == "{database}" and {schema_column} == "{schema}" and {table_column} == "{table}"'
                    )
                    doc = f"The following columns are in the {table} table in the {database} database:\n\n"
                    doc += df_columns_filtered_to_table[columns].to_markdown()

                    plan._plan.append(
                        TrainingPlanItem(
                            item_type=TrainingPlanItem.ITEM_TYPE_IS,
                            item_group=f"{database}.{schema}",
                            item_name=table,
                            item_value=doc,
                            database=table
                        )
                    )

        return plan
    
    def deterministic_uuid(self,content: Union[str, bytes]) -> str:
    
        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
        elif isinstance(content, bytes):
            content_bytes = content
        else:
            raise ValueError(f"Content type {type(content)} not supported !")

        hash_object = hashlib.sha256(content_bytes)
        hash_hex = hash_object.hexdigest()
        namespace = uuid.UUID("00000000-0000-0000-0000-000000000000")
        content_uuid = str(uuid.uuid5(namespace, hash_hex))

        return content_uuid
    
    def _dataclass_to_dict(self, obj):
        return dataclasses.asdict(obj)

    def _extract_documents(self,query_results) -> list:
        
        if query_results is None:
            return []

        if "documents" in query_results:
            documents = query_results["documents"]

            if len(documents) == 1 and isinstance(documents[0], list):
                try:
                    documents = [json.loads(doc) for doc in documents[0]]
                except Exception as e:
                    return documents[0]

            return documents
        
    def add_documentation(self, documentation: str,database: str, **kwargs) -> str:
        id = self.deterministic_uuid(documentation) + "-doc"
        documentation_json = json.dumps(
            {
                "question": None,
                "documentation": documentation,
                "database": database[0],
            },
            ensure_ascii=False,
        )

        self.documentation_collection.add(
            documents=documentation_json,
            embeddings=self.generate_embedding(documentation),
            ids=id,
            metadatas=self.convert_to_dict_list(database)
        )
        return id
    
    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        embedding = self.embedding_function([data])
        if len(embedding) == 1:
            return embedding[0]
        return embedding
    
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    def get_training_data(self, **kwargs) -> pd.DataFrame:
        sql_data = self.sql_collection.get()

        df = pd.DataFrame()

        if sql_data is not None:
            # Extract the documents and ids
            documents = [json.loads(doc) for doc in sql_data["documents"]]
            ids = sql_data["ids"]

            # Create a DataFrame
            df_sql = pd.DataFrame(
                {
                    "id": ids,
                    "question": [doc["question"] for doc in documents],
                    "content": [doc["sql"] for doc in documents],
                    "database":[doc["database"] for doc in documents],
                }
            )

            df_sql["training_data_type"] = "sql"

            df = pd.concat([df, df_sql])

        ddl_data = self.ddl_collection.get()

        if ddl_data is not None:
            # Extract the documents and ids
            documents = [doc for doc in ddl_data["documents"]]
            ids = ddl_data["ids"]

            # Create a DataFrame
            df_ddl = pd.DataFrame(
                {
                    "id": ids,
                    "question": [None for doc in documents],
                    "content": [doc["documentation"] for doc in documents],
                    "database":[doc["database"] for doc in documents],
                }
            )

            df_ddl["training_data_type"] = "ddl"

            df = pd.concat([df, df_ddl])

        doc_data = self.documentation_collection.get()

        if doc_data is not None:
            # Extract the documents and ids
            documents = [json.loads(doc) for doc in doc_data["documents"]]
            ids = doc_data["ids"]

            # Create a DataFrame
            df_doc = pd.DataFrame(
                {
                    "id": ids,
                    "question": [None for doc in documents],
                    "content": [doc["documentation"] for doc in documents],
                    "database":[doc["database"] for doc in documents],
                }
            )
            
            df_doc["training_data_type"] = "documentation"

            df = pd.concat([df, df_doc])
            
        return df

    def remove_training_data(self, id: str, **kwargs) -> bool:
        if id.endswith("-sql"):
            self.sql_collection.delete(ids=id)
            return True
        elif id.endswith("-ddl"):
            self.ddl_collection.delete(ids=id)
            return True
        elif id.endswith("-doc"):
            self.documentation_collection.delete(ids=id)
            return True
        else:
            return False
        
    def generate_question(self, sql: str, **kwargs) -> str:
        response = self.submit_prompt(
            [
                self.system_message(
                    "The user will give you SQL and you will try to guess what the business question this query is answering. Return just the question without any additional explanation. Do not reference the table name in the question."
                ),
                self.user_message(sql),
            ],
            **kwargs,
        )

        return response
#
#
#
#
#
#
#
#
#
    def train(
        self,
        question: str = None,
        sql: str = None,
        ddl: str = None,
        documentation: str = None,
        plan: TrainingPlan = None,
        database: str  =None
    ) -> str:
        database=[database]
        if question and not sql:
            raise ValidationError(f"Please also provide a SQL query")

        if documentation:
            print("Adding documentation....")
            return self.add_documentation(documentation)

        if sql:
            if question is None:
                question = self.generate_question(sql)
                print("Question generated with sql:", question, "\nAdding SQL...")
            return self.add_question_sql(question=question, sql=sql,database= database)

        if ddl:
            print("Adding ddl:", ddl)
            return self.add_ddl(ddl)

        if plan:
            for item in plan._plan:
                item.database=[item.database]
                if item.item_type == TrainingPlanItem.ITEM_TYPE_DDL:
                    self.add_ddl(item.item_value,database=item.database)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_IS:
                    self.add_documentation(item.item_value,database=item.database)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_SQL:
                    self.add_question_sql(question=item.item_name, sql=item.item_value,database=item.database)


    def generate_sql(self,question: str, **kwargs) -> str:

    
        initial_prompt = None
        # print("\ninitial prompt:",initial_prompt)
        question_sql_list = self.get_similar_question_sql(question, **kwargs)
        # print("\nquestion_sql_list:",question_sql_list)
        ddl_list = self.get_related_ddl(question, **kwargs)
        # print("\nddl_list:",ddl_list)
        doc_list = self.get_related_documentation(question, **kwargs)
        # print("\ndoc list:",doc_list)
        prompt = self.get_sql_prompt(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            **kwargs,
        )

        prompt1=""
        for i in range(0,len(prompt)):
            prompt1=prompt1+"\n"+prompt[i]
        
        prompt1=prompt1+("\nNote: 1) The revenue figures in the table are in currency: Rupees\n"
        "2) Ignore 'calculated_revenue' column if user inputs 'revenue'\n"
        "3) Convert data types of columns containing numbers other than date(like 2022-06-21) to Float.\n\n"
        "For any usage of date use datetime format which specified the date in the YYYY-MM-DD format.\n"
        "Always use schema.table_name in the query instead of just the table name"
        "For example, if the question mentions '1st january 2024' use WHERE date='2024-01-01' instead of WHERE date='1st january 2024'.\n"
        "Return only the sql query.do not add text in the output or explain the query.\n"
        "Return only the final sql query and DO NOT repeat the question.-- just the query and make sure to end with a semicolon\n"
        f"The tables in the query database are named as follows- use only these table names in the sql query-'{self.table_name}'")
        self.log(prompt1)
        llm_response = self.submit_prompt_codellama(prompt1, **kwargs)
        # self.log(f"codellama output :{llm_response}")
        return self.extract_sql(llm_response)
    
    def get_sql_prompt(
        self,
        initial_prompt : str,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ):
        
        if initial_prompt is None:
            initial_prompt = (
            "The user provides a question and you provide SQL.\n"

            "You will only respond with SQL code and not with any explanations.\n"
            
            "provide SQL queries for postgres database only.\n"
            
            "Respond with only SQL code.Do not answer with any explanations -- just the code.\n")
        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=14000
        )
        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=14000
        )
        message_log = [initial_prompt]
        message_log.append("The following are example questions along with their sql queries:")
        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    sql=example['sql'].format(table_name=self.table_name)
                    message_log.append(example["question"])
                    message_log.append(sql)
        message_log.append("The question you must answer is as follows: "+question+"\nAnswer only the last question. Answer only the last question and use the previous questions and their following queries for reference only")
        return message_log

    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        return self._extract_documents(
            self.sql_collection.query(
                query_texts=[question],
                n_results=self.n_results,
                where={'database': self.db_being_used},
            )
        )
    
    def get_related_ddl(self, question: str, **kwargs) -> list:
        temp_ddl_list=self._extract_documents(
            self.ddl_collection.query(
                query_texts=[question],
                where={'database': self.db_being_used},
            )
        )
        ddl_list=[]
        for example in temp_ddl_list:
            ddl_list.append(example["documentation"])
        return ddl_list
    
    def add_ddl_to_prompt(
        self, initial_prompt: str, ddl_list: List[str], max_tokens: int = 14000
    ) -> str:
        if len(ddl_list) > 0:
            initial_prompt += f"\nYou may use the following DDL statements as a reference for what tables might be available. Use responses to past questions also to guide you:\n\n"

            for ddl in ddl_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(ddl)
                    < max_tokens
                ):
                    initial_prompt += f"{ddl}\n\n"

        return initial_prompt

    def get_related_documentation(self, question: str, **kwargs) -> list:
        temp_doc_list=self._extract_documents(
            self.documentation_collection.query(
                query_texts=[question],
                where={'database': self.db_being_used}
            )
        )
        doc_list=[]
        for example in temp_doc_list:
            doc_list.append(example["documentation"])
        return doc_list
    
    def add_documentation_to_prompt(
        self,
        initial_prompt: str,
        documentation_list: List[str],
        max_tokens: int = 14000,
    ) -> str:
        if len(documentation_list) > 0:
            initial_prompt += f"\nYou may use the following documentation as a reference for what tables might be available. Use responses to past questions also to guide you:\n\n"

            for documentation in documentation_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(documentation)
                    < max_tokens
                ):
                    initial_prompt += f"{documentation}\n\n"

        return initial_prompt

    def run_sql(self,sql: str) -> Union[pd.DataFrame, None]:
        conn=self.conn
        if conn is not False:
            try:
                cs = conn.cursor()
                cs.execute(sql)
                results = cs.fetchall()

                # Create a pandas dataframe from the results
                df = pd.DataFrame(
                    results, columns=[desc[0] for desc in cs.description]
                )
                return df

            except psycopg2.Error as e:
                conn.rollback()
                raise ValidationError(e)

            except Exception as e:
                conn.rollback()
                raise e

    def add_ddl(self, ddl: str,database: str, **kwargs) -> str:
        ddl_json = json.dumps(
            {
                "question": None,
                "documentation": ddl,
                "database": database[0],
            },
            ensure_ascii=False,
        )
        id = self.deterministic_uuid(ddl) + "-ddl"
        self.ddl_collection.add(
            documents=ddl_json,
            embeddings=self.generate_embedding(ddl),
            ids=id,
            metadatas=self.convert_to_dict_list(database),
        )
        return id

    def add_question_sql(self, question: str, sql: str,database: str, **kwargs) -> str:
        question_sql_json = json.dumps(
            {
                "question": question,
                "sql": sql,
                "database": database[0],
            },
            ensure_ascii=False,
        )

        id = self.deterministic_uuid(question_sql_json) + "-sql"
        self.sql_collection.add(
            documents=question_sql_json,
            embeddings=self.generate_embedding(question_sql_json),
            ids=id,
            metadatas=self.convert_to_dict_list(database)
        )

        return id
    
    def _sanitize_plotly_code(self, raw_plotly_code: str) -> str:
        # Remove the fig.show() statement from the plotly code
        plotly_code = raw_plotly_code.replace("fig.show()", "")

        return plotly_code
    
    def generate_plotly_code(
        self, question: str = None, sql: str = None, df: str = None, **kwargs
    ) -> str:
        if question is not None:
            system_msg = f"The following is a pandas DataFrame that contains the results of the query that answers the question the user asked: '{question}'"
        else:
            system_msg = "The following is a pandas DataFrame "
        
        df_metadata=self.df_to_csv_string(df)
        
        if sql is not None:
            system_msg += f"\n\nThe DataFrame was produced using this query: {sql}\n\n"

        system_msg += f"""The following is information about the resulting pandas DataFrame 'df' in csv format : \n{df_metadata}"
        Can you generate the Python plotly code to chart the results of the dataframe? 
        Assume the data is in a pandas dataframe called 'df'. The data in cs format might have index included, ignore index values .
        Ensure that the x-axis and y-axis autoscales based on the range of x and y values.If there is only one value in the dataframe, use an Indicator. 
        Consider utilizing multiple plots on the graph for different identifiers when the dataframe comprises multiple columns. 
        Respond with only Python code. Do not answer with any explanations -- just the code."""
        
        message_log=system_msg
        # self.log("plotly prompt: ",message_log)

        plotly_code = self.submit_prompt_codellama(message_log, kwargs=kwargs)
        return self._sanitize_plotly_code(self._extract_python_code(plotly_code))

    def get_plotly_figure(
        self, plotly_code: str, df: pd.DataFrame, dark_mode: bool = True
    ) -> plotly.graph_objs.Figure:
        
        ldict = {"df": df, "px": px, "go": go}
        try:
            exec(plotly_code, globals(), ldict)

            fig = ldict.get("fig", None)
        except Exception as e:
            # Inspect data types
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            # Decision-making for plot type
            if len(numeric_cols) >= 2:
                # Use the first two numeric columns for a scatter plot
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
            elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
                # Use a bar plot if there's one numeric and one categorical column
                fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0])
            elif len(categorical_cols) >= 1 and df[categorical_cols[0]].nunique() < 10:
                # Use a pie chart for categorical data with fewer unique values
                fig = px.pie(df, names=categorical_cols[0])
            else:
                # Default to a simple line plot if above conditions are not met
                fig = px.line(df)

        if fig is None:
            return None

        if dark_mode:
            fig.update_layout(template="plotly_dark")

        return fig

    def ask(
        self,
        question: Union[str, None] = None,
        print_results: bool = True,
        auto_train: bool = True,
        result_summary: bool = True,
        visualize: bool = True,
        username=None,
          # if False, will not generate plotly code
    ) -> Union[
        Tuple[
            Union[str, None],
            Union[pd.DataFrame, None],
            Union[plotly.graph_objs.Figure, None],
        ],
        None,
    ]:
        if question is None:
            question = input("Enter a question: ")

        self.add_user_question(question,username)

        try:
            sql = self.generate_sql(question=question)
        except Exception as e:
            print(e)
            return None, None, None, None

        if print_results:
            try:
                Code = __import__("IPython.display", fromList=["Code"]).Code
                display(Code(sql))
            except Exception as e:
                print("query:\n", sql)

        if self.run_sql_is_set is False:
            print(
                "If you want to run the SQL query, connect to a database first."
            )

            if print_results:
                return None
            else:
                return sql, None, None, None
        try:
            df = self.run_sql(sql)
            if print_results:
                try:
                    display = __import__(
                        "IPython.display", fromList=["display"]
                    ).display
                    display(df)
                except Exception as e:
                    print("The retrieved data :\n",df)
            if len(df) > 0 and auto_train:
                self.add_question_sql(question=question, sql=sql)
            if result_summary:
                try:
                    text_summary= self.generate_text_summary(
                        question=question,
                        sql=sql,
                        df=df,
                    )
            
                    if print_results:
                        try:
                            print("text_summary : \n",text_summary)
                        except Exception as e:
                            print("Error printing summary")
                except Exception as e:
                    # Print stack trace
                    traceback.print_exc()
                    print("Couldn't generate text summary: ", e)
                    if print_results:
                        return None
                    else:
                        return sql, df, None, None
            else:
                return sql, df, None, None
            if visualize:
                try:
                    plotly_code = self.generate_plotly_code(
                        question=question,
                        sql=sql,
                        df=df,
                    )
                    fig = self.get_plotly_figure(plotly_code=plotly_code, df=df)
                    img_bytes = fig.to_image(format="png",scale=2)
                    img = Image.open(BytesIO(img_bytes))
                    # img=fig
                    plot=img

                    if print_results:
                        try:
                            img.show()
                        except:
                            print("error displaying image")

                except Exception as e:
                    # Print stack trace
                    traceback.print_exc()
                    print("Couldn't run plotly code: ", e)
                    if print_results:
                        return None
                    else:
                        return sql, df, text_summary, None
            else:
                return sql, df, text_summary, None
        
        except Exception as e:
            print("Couldn't run sql: ", e)
            if print_results:
                return None
            else:
                return sql, None, None, None
        return sql,df,text_summary,plot

    def str_to_approx_token_count(self, string: str) -> int:
        return len(string) / 4
    
    def system_message(self, message: str) -> any:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> any:
        return {"role": "user", "content": message}

    def log(self, message: str):
        print(message)

    def submit_prompt(self, prompt, **kwargs) -> str:
        # print("\n\nContext length of prompt :",len(prompt),"\n")
    
        ollama = Ollama(base_url=config.ollama_host, model=config.text_model)

        results=ollama(prompt)

        return results
    
    def submit_prompt_codellama(self, prompt, **kwargs) -> str:
        # print("\n\nContext length of prompt :",len(prompt),"\n")

        ollama = Ollama(base_url=config.ollama_host, model=config.code_model)

        results=ollama(prompt)

        return results

    def extract_sql(self, llm_response: str) -> str:
        # If the llm_response contains a markdown code block, with or without the sql tag, extract the sql from it
        sql = re.search(r"```sql\n(.*)```", llm_response, re.DOTALL)
        if sql:
            extracted_sql = sql.strip()
            if not extracted_sql.endswith(";"):
                extracted_sql += ";"
            # self.log(f"Output from LLM: {llm_response} \nExtracted SQL: {sql.group(1)}")
            return extracted_sql

        sql = re.search(r"```(.*)```", llm_response, re.DOTALL)
        if sql:
            extracted_sql = sql.strip()
            if not extracted_sql.endswith(";"):
                extracted_sql += ";"# self.log(f"Output from LLM: {llm_response} \nExtracted SQL: {sql.group(1)}")
            return extracted_sql
        
        pattern = re.compile(re.escape("SELECT") + r'.*?' + re.escape(";"), re.DOTALL)
        llm_response = pattern.findall(llm_response)[0]
        if llm_response:
            extracted_sql = llm_response.strip()
            if not extracted_sql.endswith(";"):
                llm_response += ";"
        
            return llm_response
    
    def _extract_python_code(self, markdown_string: str) -> str:
        # Regex pattern to match Python code blocks
        pattern = r"```[\w\s]*python\n([\s\S]*?)```|```([\s\S]*?)```"

        # Find all matches in the markdown string
        matches = re.findall(pattern, markdown_string, re.IGNORECASE)

        # Extract the Python code from the matches
        python_code = []
        for match in matches:
            python = match[0] if match[0] else match[1]
            python_code.append(python.strip())

        if len(python_code) == 0:
            return markdown_string

        return python_code[0]
    
    def generate_text_summary(
    self, question: str = None, sql: str = None, df: str = None, **kwargs
    ) -> str:
        if question is not None:
            system_msg = f"The following is a pandas DataFrame that contains the results of the query that answers the question the user asked: '{question}'"
        else:
            system_msg = "The following is a pandas DataFrame "

        if sql is not None:
            system_msg += f"\n\nThe DataFrame was produced using this query: {sql}\n\n"
        
        df_metadata = self.df_to_csv_string(df)

        system_msg += f"""The following is information about the resulting pandas DataFrame 'df' in csv format: \n{df_metadata}.
        Can you generate the summary of the results of the dataframe? Assume the data is in a pandas dataframe called 'df'. 
        Do not make assumptions that are not referenced in the prompt. Make small assumptions if required, for example- when representing days of the week u can assume sunday instead of 0 and so on.
        Respond with only text summary along with the explanation for the numbers in the answer. 
        If the df refrences to ectent then extent is in meters. Do not generate new explanations. Just respond with the neccessary.
        Do not answer by repeating the question. -- just the text summary along with the output numbers. 
        Do not mention or quote the sql query or df name in the response.Only Explain the df in natural language."""
        message_log = system_msg
        # print("shape of message log :", len(message_log))
        # print("message log:", message_log)
        text_summary = self.submit_prompt(message_log, kwargs=kwargs)
        return text_summary
    
    def df_to_csv_string(self,df):

        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()
        csv_buffer.close()

        return csv_string
    
    def convert_to_dict_list(self,strings_list):
        dict_list = []
        for string in strings_list:
            dict_list.append({"database": string})
        return dict_list
    
    def main_setup(self):
        
        DB_NAME=config.DB_NAME
        DB_USER=config.DB_USER
        DB_PASSWORD=config.DB_PASSWORD
        DB_PORT=config.DB_PORT
        HOST=config.HOST
        path_congestion=config.path_congestion
        path_trafficflow=config.path_trafficflow
        tables=config.tables_in_db

        if tables:
            df_information_schema=[]
            for i in range(0,len(tables)):
                df_information_schema_temp = self.run_initial_sql(f"SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE table_schema='{tables[i].split('.')[0]}' AND table_name='{tables[i].split('.')[1]}'",DB_NAME,DB_USER,DB_PASSWORD,DB_PORT,HOST)
                df_information_schema.append(df_information_schema_temp)
        print("data retrieved successfully")
        plan_array=[]
        for i in range(0,len(df_information_schema)):
            plan = self.get_training_plan_generic(df_information_schema[i])
            plan_array.append(plan)        
        print("plan created") 

        questions_ans_database_congestion=np.load(path_congestion)
        questions_congestion=questions_ans_database_congestion[:,0].tolist()
        ans_congestion=questions_ans_database_congestion[:,1].tolist()
        database_congestion=questions_ans_database_congestion[:,2].tolist()

        questions_ans_database_trafficflow=np.load(path_trafficflow)
        questions_trafficflow=questions_ans_database_trafficflow[:,0].tolist()
        ans_trafficflow=questions_ans_database_trafficflow[:,1].tolist()
        database_trafficflow=questions_ans_database_trafficflow[:,2].tolist()

        if len(self.get_training_data()[self.get_training_data()["training_data_type"] =="sql"]) != (len(questions_congestion)+len(questions_trafficflow)) or len(self.get_training_data()[self.get_training_data()["training_data_type"] =="documentation"].index) != len(plan_array):            
            print("\n\nThe training data is being updated\n\n")

            #remove/empty training data
            for i in self.get_training_data()["id"]:
                self.remove_training_data(i)
                print("training data removed successfully")
            
            #training
            for i in plan_array:
                self.train(plan=i)  # Add documentation items
                print("added table : ", i)


            print(tables[0],tables[1])    
            i=0
            for i in range(len(questions_congestion)):
                self.train(question=questions_congestion[i],sql=ans_congestion[i],database=database_congestion[i])  # Add sql question answer pairs
                i=i+1
            print(f"added {i} congestion questions")
            i=0
            for i in range(len(questions_trafficflow)):
                self.train(question=questions_trafficflow[i],sql=ans_trafficflow[i],database=database_trafficflow[i])  # Add sql question answer pairs
                i=i+1
            print(f"added {i} trafficflow questions")
            print("training finished")

        else:
            print("\n\nThe traning data is the same as previous iteration\nSkiping training\n\n")


        training_data =self.get_training_data()
        print("training data: \n",training_data)

    def ask_function_in_loop(self,print_results,auto_train,result_summary,visualize):
        user_input=""
        while user_input!='q':
            user_input=input("Type your question:\n")
            start_time=time.time()

            self.ask(question=user_input,
                            print_results = print_results,
                            auto_train = auto_train,
                            result_summary = result_summary,
                            visualize = visualize)
            print("end of process!")

            end_time=time.time()

            run_time=end_time- start_time
            print("run time : ",run_time)
        training_data =self.get_training_data()
        print("training data: ",training_data)

    def get_user_questions(self, **kwargs) -> pd.DataFrame:
        user_questions= self.user_questions_storage_collection.get()
        df=pd.DataFrame()

        if user_questions is not None:
            documents= [json.loads(doc) for doc in user_questions["documents"]]
            ids=user_questions["ids"]
        df_user_questions = pd.DataFrame(
                {
                    "id": ids,
                    "timestamp": [doc["timestamp"] for doc in documents],
                    "user": [doc["user"] for doc in documents],
                    "question":[doc["question"] for doc in documents],
                    "database":[doc["database"] for doc in documents]
                }
            )
        df = pd.concat([df, df_user_questions])
        return df
    
    def remove_user_questions(self,id: str, **kwargs) -> bool:
        if id.endswith("-user_question"):
            self.user_questions_storage_collection.delete(ids=id)
            return True
        else:
            return False
        
    def add_user_question(self,question: str,username,**kwargs) -> str:

        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        user_questions_json = json.dumps(
            {
                "timestamp": current_timestamp,
                "user":username,
                "question":question,
                "database":self.db_being_used,
            },
            ensure_ascii=False,
        )

        id=self.deterministic_uuid(user_questions_json) + "-user_question"
        self.user_questions_storage_collection.add(
            documents=user_questions_json,
            embeddings=self.generate_embedding(user_questions_json),
            ids=id,
        )
