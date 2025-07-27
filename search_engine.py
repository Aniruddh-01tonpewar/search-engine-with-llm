
import streamlit as st
from  langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType

from langchain_community.callbacks.streamlit import StreamlitCallbackHandler


ArxivWrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
WikiWrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)

Arxiv = ArxivQueryRun(api_wrapper=ArxivWrapper)
Wiki = WikipediaQueryRun(api_wrapper=WikiWrapper)
DDGSWrapper = DuckDuckGoSearchAPIWrapper(max_results=1)
Search = DuckDuckGoSearchRun(api_wrapper=DDGSWrapper)

st.title("Search Engine with the agent")
st.sidebar.title("Enter Groq API Key")
api_key=st.sidebar.text_input('click here to enter API key',type="password")

if "messages" not in st.session_state:
    st.session_state['messages']=[
        {'role':"assistant",'content':"Hi, I am a chatbot who can search the web. How can I help you?"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt:= st.chat_input(placeholder="What is machine learning"):
    st.session_state.messages.append({'role':'user','content':prompt})
    st.chat_message('user').write(prompt)   

    llm = ChatGroq(api_key=api_key,model_name='gemma2-9b-it',streaming=True)
    tools = [ Arxiv, Wiki, Search] 

    search_agent = initialize_agent(tools,llm, agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_erros= True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant','content':response})
        
        st.chat_message("assistant").write(response)