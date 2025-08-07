import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from agents.agent import rag_agent_executor, memory
import datetime
import asyncio
from streamlit import dialog as st_dialog
import time

# Enable detailed error messages in the Streamlit client console
st.set_option("client.showErrorDetails", True)

# Define the template for the assistant's prompt with current time and user question
PROMPT_TEMPLATE = """
    Give a detailed answer to the question and explain the answer clearly, if you don't know the answer to the question, even after using the provided tool, state that you don't know the answer. 
    Provide sources section if possible.
    Current Date and Time: {current_time}
    question: {question}
"""
# Error recovery function; displays error and offers session reset
@st_dialog("Error Recovery")
def error_fallback(e):
    st.error(f"An error has occured: {e}")
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()

# Main asynchronous function for handling the chat interface and responses
def main():
    # Configure page title and icon, and display the main title
    st.set_page_config(page_title="LinkMind", page_icon=":link::brain:")
    st.title(":link::brain: LinkMind")       

    # Get user's chat input
    prompt = st.chat_input("Message LinkMind...")

    # Initialize session history for messages if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages (excluding system messages)
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt:
        # Show the user's new message and add it to session history
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate and display assistant response with a loading spinner
        with st.chat_message("assistant", avatar="🤖"), st.empty():
            with st.spinner("Thinking...", show_time=True):
                try:
                    # Format the prompt with current date/time and user question
                    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                    formatted_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    query = prompt_template.format(current_time=formatted_now, question=prompt)

                    # Invoke the agent asynchronously using version "v2"
                    start=time.time();
                    response = rag_agent_executor.invoke({"input": query}, version="v2")
                    total_time=round(time.time()-start);
                    output_text = f"*Thought for {total_time} seconds*\n\n"+response["output"]
                except Exception as e:
                    # On error, use fallback mechanism and set output_text to the error
                    output_text = e
                    error_fallback(e)

            # Display the assistant's response and save it to session history
            st.markdown(output_text)
            st.session_state.messages.append({"role": "assistant", "content": output_text})

# Run the main async function if this script is executed as the main module
if __name__ == "__main__":
    main()
