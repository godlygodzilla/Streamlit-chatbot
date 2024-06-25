import streamlit as st
import requests
import json
import base64
from io import BytesIO
from PIL import Image
import pandas as pd

# Function to get a response from the Django backend
def get_response(user_input,show_plot):
    url = 'http://square-martin-obliging.ngrok-free.app/chat/chatbot/'
    headers = {'Content-Type': 'application/json'}
    payload = {
        'username': 'example_user',
        'mode': 'ASK',
        'question': user_input,
        'table_key': 'congestion',
        'show_plot': show_plot,
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        data = response.json().get('data', {})
        return data.get('sql', 'No SQL query generated'), data.get('df', 'No data frame generated'), data.get('text_summary', 'No summary generated'), data.get('plot', 'No plot generated')
    else:
        return None, None, None, None
    
def display_plot(plot_base64):
    if plot_base64:
        plot_data = base64.b64decode(plot_base64)
        plot_image = Image.open(BytesIO(plot_data))
        st.image(plot_image)

# Streamlit app
st.title("Chatbot")

if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

with st.form(key='chat_form'):
    user_input = st.text_input("You: ", key='user_input')
    submit_button = st.form_submit_button(label='Send')

show_text_summary= st.checkbox("text_summary", value=True)
show_plot = st.checkbox("Plot", value=False)

if submit_button and user_input:
    sql, df, text_summary, plot = get_response(user_input,show_plot)
    df = df.to_dict(orient='records') if isinstance(df, pd.DataFrame) else df

    print("sql   ",sql,"\ndf   ",df,"\n summary   ",text_summary,"\n plot    ",plot)
    st.session_state.conversation.append({
        "user_input": user_input,
        "sql": sql,
        "df": df,
        "text_summary": text_summary,
        "plot": plot
    })

for entry in st.session_state.conversation:
    st.markdown(f"<b style='color:blue;'>You:</b> {entry['user_input']}", unsafe_allow_html=True)
    if entry['sql']:
        st.write(f"**SQL Query:**\n {entry['sql']}")
    if entry['df']:
        st.write("**Data Frame:**")
        st.dataframe(entry['df'])  # Display the DataFrame using st.dataframe
    if entry['text_summary']:
        st.write(f"**Summary:**\n{entry['text_summary']}")
    if entry['plot']:
        st.write("**Plot:**")
        display_plot(entry['plot'])  # Display the plot
