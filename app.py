import streamlit as st
import os
from groq import Groq
from serpapi import GoogleSearch
from deep_translator import GoogleTranslator
from langdetect import detect

# Streamlit app title and description
st.title("Enhanced Mini Perplexity - Now with Thanglish Support!")
st.write("Ask me anything in English or Thanglish, and I'll generate a response using the LLaMA model and search the web for up-to-date information!")

# Get the API keys from environment variables
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
SERPAPI_API_KEY = st.secrets["SERPAPI_API_KEY"]

# Initialize Groq client and Google Translator
client = Groq(api_key=GROQ_API_KEY)
translator = GoogleTranslator(source='auto', target='en')

# Initialize session state to keep track of chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def translate_if_needed(text):
    try:
        # Detect language
        detected_lang = detect(text)
        if detected_lang != 'en':
            return translator.translate(text)
        return text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

def search_web(query):
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": 5
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    
    if "organic_results" in results:
        return results["organic_results"][:5]
    else:
        return []

def format_search_results(results):
    formatted_results = []
    for result in results:
        formatted_results.append(f"Title: {result['title']}\nSnippet: {result['snippet']}\nLink: {result['link']}\n")
    return "\n".join(formatted_results)

def call_llama_groq_api(prompt, include_web_search=False):
    try:
        translated_prompt = translate_if_needed(prompt)
        
        if include_web_search:
            search_results = search_web(translated_prompt)
            formatted_results = format_search_results(search_results)
            enhanced_prompt = f"Based on the following web search results and your knowledge, please answer the question: '{translated_prompt}'\n\nWeb search results:\n{formatted_results}\n\nYour response:"
        else:
            enhanced_prompt = translated_prompt
            search_results = []

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": enhanced_prompt,
                }
            ],
            model="llama-3.1-70b-versatile",
            max_tokens=500,
            temperature=0.7,
            top_p=0.9
        )
        return chat_completion.choices[0].message.content, search_results
    except Exception as e:
        return f"Error: {str(e)}", []

# Input field for the user's question
user_input = st.text_input("Enter your question (in English or Thanglish):")

# Checkbox to enable/disable web search
use_web_search = st.checkbox("Enable web search for up-to-date information")

# When the user submits a question
if user_input:
    with st.spinner("Generating response..."):
        response_text, search_results = call_llama_groq_api(user_input, include_web_search=use_web_search)
        
        # Display the AI-generated response
        st.subheader("AI Response:")
        st.write(response_text)
        
        # Display web search results if enabled
        if use_web_search and search_results:
            st.subheader("Web Search Results:")
            for idx, result in enumerate(search_results, 1):
                with st.expander(f"Source {idx}: {result['title']}"):
                    st.write(f"**Snippet:** {result['snippet']}")
                    st.write(f"**Link:** {result['link']}")
        
        # Add the new question and response to the chat history
        st.session_state.chat_history.append({
            "question": user_input,
            "response": response_text,
            "web_results": search_results if use_web_search else []
        })
        st.success("Response generated!")

# Display the chat history
if st.session_state.chat_history:
    st.write("### Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"**You:** {chat['question']}")
        st.write(f"**Assistant:** {chat['response']}")
        if chat['web_results']:
            st.write("**Web Sources:**")
            for idx, result in enumerate(chat['web_results'], 1):
                st.write(f"{idx}. [{result['title']}]({result['link']})")
        st.write("---")  # Separator between questions
