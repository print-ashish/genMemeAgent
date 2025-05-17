# Import relevant functionality
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from typing import Annotated, List
from pydantic import StringConstraints, Field
from pydantic import BaseModel

import os

# Import relevant functionality
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
import os

# Set the Groq API key
os.environ["GROQ_API_KEY"] = "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"

from groq import Groq
import base64


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# image_path = r"c:\Users\pc\Downloads\just rlly smart.jpg"
# image_path  = r"c:\Users\pc\Downloads\Silly milly that’s your clone!!!Get a clone for YOUR pet by following the link in @softcatmemes bio!!!#fatfatmillycat.jpg"

image_path = r"c:\Users\pc\Downloads\do you fw me.jpg"

# image_path = r"c:\Users\pc\Downloads\everytime.jpg"

# image_path = r"c:\Users\pc\Downloads\I lied..jpg"
# Getting the base64 string
base64_image = encode_image(image_path)

client = Groq(api_key="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")
# Define the greet tool
@tool
def extract_text() -> str:
    """extract the text from the image"""


    prompt ="""Extract the meme text from the image
    """
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




# Define the response model for five recipe names
class MemeTextResponseList(BaseModel):
    meme_list: Annotated[
        List[str],
        Field(min_items=5, max_items=5, description="Exactly five meme texts")
    ] = Field(..., description="List of meme texts")


# Create the agent
model = ChatGroq(temperature=0.7, model_name="llama-3.3-70b-versatile",)
tools = [extract_text]
agent_executor = create_react_agent(model, tools , response_format=MemeTextResponseList)

# Run the agent with a sample input
input_query = "You are a text extractor from the image using the tool and return the  similar meme from that meme text vibe without any explaination just give 5 meme text related to that vibe "



final_prompt = input_query 

result =  agent_executor.invoke({"messages": [HumanMessage(content=final_prompt)]})


structured_response = result["structured_response"]

# Get the list of recipe names
memes = structured_response.meme_list
print(memes)
