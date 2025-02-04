import streamlit as st
import os
import openai  # Import OpenAI library
from serpapi import GoogleSearch
from deep_translator import GoogleTranslator
from langdetect import detect
import re

# Streamlit app title and description
st.title("Enhanced Mini Perplexity - Advanced Thanglish Support!")
st.write("Ask me anything in English or Thanglish, and I'll generate a response using advanced NLP techniques and up-to-date web information!")

# Get the API keys from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
SERPAPI_API_KEY = st.secrets["SERPAPI_API_KEY"]

# Set your OpenAI API key
openai.api_key = OPENAI_API_KEY

translator = GoogleTranslator(source='auto', target='en')

# Initialize session state to keep track of chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def is_thanglish(text):
    """Detect if the text is in Thanglish (Tamil+English mix)."""
    tamil_chars = re.findall(r'[\u0B80-\u0BFF]', text)  # Tamil Unicode range
    english_chars = re.findall(r'[a-zA-Z]', text)
    return bool(tamil_chars) and bool(english_chars)

def translate_if_needed(text):
    try:
        detected_lang = detect(text)
        if detected_lang == 'ta':
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
    return results.get("organic_results", [])[:5]

def format_search_results(results):
    formatted_results = [
        f"Title: {result['title']}\nSnippet: {result['snippet']}\nLink: {result['link']}\n"
        for result in results
    ]
    return "\n".join(formatted_results)

def call_gpt4_api(prompt, include_web_search=False):
    try:
        # Determine if the prompt is Thanglish and adjust accordingly
        if is_thanglish(prompt):
            enhanced_prompt = (
                f"Respond in Thanglish naturally with conversational fluency. "
                f"Ensure casual, engaging, and contextually relevant replies. "
                f"User query: {prompt}"
            )
        else:
            translated_prompt = translate_if_needed(prompt)
            if include_web_search:
                search_results = search_web(translated_prompt)
                formatted_results = format_search_results(search_results)
                enhanced_prompt = (
                    f"Based on the web search results and your internal knowledge, answer the question: "
                    f"'{translated_prompt}'\n\nWeb search results:\n{formatted_results}\n\nYour response:"
                )
            else:
                enhanced_prompt = translate_if_needed(prompt)
                search_results = []

        # Call the OpenAI GPT-4 API
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Specify the GPT-4 model
            messages=[{"role": "user", "content": enhanced_prompt}],
            max_tokens=500,
            temperature=0.7,
            top_p=0.9
        )
        return response.choices[0].message['content'], search_results
    except Exception as e:
        return f"Error: {str(e)}", []

# User input and UI components
user_input = st.text_input("Enter your question (in English or Thanglish):")
use_web_search = st.checkbox("Enable web search for up-to-date information")

if user_input:
    with st.spinner("Generating response..."):
        response_text, search_results = call_gpt4_api(user_input, include_web_search=use_web_search)
        st.subheader("AI Response:")
        st.write(response_text)
        
        if use_web_search and search_results:
            st.subheader("Web Search Results:")
            for idx, result in enumerate(search_results, 1):
                with st.expander(f"Source {idx}: {result['title']}"):
                    st.write(f"**Snippet:** {result['snippet']}")
                    st.write(f"**Link:** {result['link']}")
        
        st.session_state.chat_history.append({
            "question": user_input,
            "response": response_text,
            "web_results": search_results if use_web_search else []
        })
        st.success("Response generated!")

if st.session_state.chat_history:
    st.write("### Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"**You:** {chat['question']}")
        st.write(f"**Assistant:** {chat['response']}")
        if chat['web_results']:
            st.write("**Web Sources:**")
            for idx, result in enumerate(chat['web_results'], 1):
                st.write(f"{idx}. [{result['title']}]({result['link']})")
        st.write("---")
