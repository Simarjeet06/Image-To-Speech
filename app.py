# import os
# import time
# from typing import Any

# import requests
# import streamlit as st
# from dotenv import find_dotenv, load_dotenv
# from transformers import pipeline

# # ⭐ Correct LangChain imports
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.chat_models import ChatOllama

# from utils.custom import css_code

# load_dotenv(find_dotenv())
# HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")


# def progress_bar(amount_of_time: int) -> Any:
#     """A simple progress bar that increases over time"""
#     progress_text = "Please wait, Generative models hard at work"
#     my_bar = st.progress(0, text=progress_text)

#     for percent_complete in range(amount_of_time):
#         time.sleep(0.04)
#         my_bar.progress(percent_complete + 1, text=progress_text)
#     time.sleep(1)
#     my_bar.empty()


# def generate_text_from_image(url: str) -> str:
#     """Generate a caption for an image using BLIP model"""
#     image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
#     generated_text: str = image_to_text(url)[0]["generated_text"]

#     print(f"IMAGE INPUT: {url}")
#     print(f"GENERATED TEXT OUTPUT: {generated_text}")
#     return generated_text


# def generate_story_from_text(scenario: str) -> str:
#     """Generate a 50-word short story from image caption using Ollama (LLaMA3)"""

#     prompt = PromptTemplate(
#         input_variables=["scenario"],
#         template="""
#         You are a talented storyteller who can create a story from a simple narrative.
#         Create a story using the following scenario; the story must be maximum 50 words.

#         CONTEXT: {scenario}
#         STORY:
#         """,
#     )

#     llm = ChatOllama(model="llama3", temperature=0.9)

#     chain = prompt | llm | StrOutputParser()

#     generated_story = chain.invoke({"scenario": scenario})

#     print(f"TEXT INPUT: {scenario}")
#     print(f"GENERATED STORY OUTPUT: {generated_story}")
#     return generated_story


# def generate_speech_from_text(message: str) -> Any:
#     """Generate speech audio file from text using HuggingFace TTS"""
#     API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
#     headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
#     payload = {"inputs": message}

#     response = requests.post(API_URL, headers=headers, json=payload)

#     with open("generated_audio.flac", "wb") as file:
#         file.write(response.content)


# def main() -> None:
#     st.set_page_config(page_title="IMAGE TO STORY CONVERTER", page_icon="🖼️")

#     st.markdown(css_code, unsafe_allow_html=True)

#     with st.sidebar:
#         st.image("img/gkj.jpg")
#         st.write("---")
#         st.write("AI App")

#     st.header("Image-to-Story Converter")

#     uploaded_file = st.file_uploader("Please choose a file to upload", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         bytes_data = uploaded_file.getvalue()

#         with open(uploaded_file.name, "wb") as file:
#             file.write(bytes_data)

#         st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

#         progress_bar(100)

#         scenario = generate_text_from_image(uploaded_file.name)
#         story = generate_story_from_text(scenario)
#         generate_speech_from_text(story)

#         with st.expander("Generated Image Scenario"):
#             st.write(scenario)

#         with st.expander("Generated Short Story"):
#             st.write(story)

#         st.audio("generated_audio.flac")


# if __name__ == "__main__":
#     main()

import os
import time
from typing import Any

import requests
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

from utils.custom import css_code

# Load environment variables
load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")


def progress_bar(amount_of_time: int) -> Any:
    progress_text = "Please wait, Generative models hard at work"
    bar = st.progress(0, text=progress_text)

    for percent in range(amount_of_time):
        time.sleep(0.02)
        bar.progress(percent + 1, text=progress_text)

    time.sleep(0.5)
    bar.empty()


def generate_text_from_image(url: str) -> str:
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    caption = image_to_text(url)[0]["generated_text"]

    print(f"[BLIP] Image: {url}")
    print(f"[BLIP] Caption: {caption}")
    return caption


def generate_story_from_text(scenario: str) -> str:
    prompt = PromptTemplate(
        input_variables=["scenario"],
        template="""
        Create a short story of max 50 words from this scenario:

        {scenario}

        STORY:
        """,
    )

    llm = ChatOllama(model="llama3", temperature=0.9)
    chain = prompt | llm | StrOutputParser()

    story = chain.invoke({"scenario": scenario})

    print(f"[LLM] Scenario: {scenario}")
    print(f"[LLM] Story: {story}")
    return story


from gtts import gTTS

def generate_speech_from_text(message: str) -> str:
    """
    Generates speech from text using Google Text-to-Speech (gTTS).
    Saves as 'generated_audio.mp3' and returns the filename.
    """
    audio_path = "generated_audio.mp3"
    try:
        tts = gTTS(text=message, lang="en")
        tts.save(audio_path)
        print("Audio binary received ✔️")
        print("Saved audio to", audio_path)
        return audio_path
    except Exception as e:
        print(f"[gTTS ERROR]: {e}")
        return None


def main() -> None:
    st.set_page_config(page_title="IMAGE TO STORY CONVERTER", page_icon="🖼️")
    st.markdown(css_code, unsafe_allow_html=True)

    with st.sidebar:
        st.image("img/gkj.jpg")
        st.write("---")
        st.write("AI App")

    st.header("Image-to-Story Converter")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        file_bytes = uploaded.getvalue()

        with open(uploaded.name, "wb") as f:
            f.write(file_bytes)

        st.image(uploaded, caption="Uploaded Image", use_column_width=True)

        progress_bar(80)

        caption = generate_text_from_image(uploaded.name)
        story = generate_story_from_text(caption)
        audio_file = generate_speech_from_text(story)

        with st.expander("Generated Caption"):
            st.write(caption)

        with st.expander("Generated Story"):
            st.write(story)

        # AUDIO HANDLING
        if audio_file and os.path.exists(audio_file):
            st.success("Audio successfully generated!")
            st.audio(audio_file)
        else:
            st.error("Audio generation failed. Check terminal logs.")


if __name__ == "__main__":
    main()
