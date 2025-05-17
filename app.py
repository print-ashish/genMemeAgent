import streamlit as st
from groq import Groq
import base64
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Annotated, List
import os

# Set API key
os.environ["GROQ_API_KEY"] = "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"

# Groq client
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Function to encode image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Tool to extract meme text
@tool
def extract_text() -> str:
    """extract the text from the image"""
    prompt = "Extract the meme text from the image"
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
    )
    return chat_completion.choices[0].message.content

# Response format
class MemeTextResponseList(BaseModel):
    meme_list: Annotated[
        List[str],
        Field(min_items=5, max_items=5, description="Exactly five meme texts")
    ] = Field(..., description="List of meme texts")

# Agent setup
model = ChatGroq(temperature=0.7, model_name="llama-3.3-70b-versatile")
tools = [extract_text]
agent_executor = create_react_agent(model, tools, response_format=MemeTextResponseList)

# --- Streamlit UI ---
st.title("Meme Vibe Extractor 🔍")

uploaded_file = st.file_uploader("Upload a meme image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())
    
    base64_image = encode_image("temp.jpg")
    
    input_query = "You are a text extractor from the image using the tool and return the similar meme from that meme text vibe without any explaination just give 5 meme text related to that vibe"
    final_prompt = input_query

    with st.spinner("Analyzing meme..."):
        result = agent_executor.invoke({"messages": [HumanMessage(content=final_prompt)]})
        structured_response = result["structured_response"]
        memes = structured_response.meme_list

    st.subheader("🎯 Extracted Vibe Memes")
    for idx, meme in enumerate(memes, 1):
        st.markdown(f"**{idx}.** {meme}")
