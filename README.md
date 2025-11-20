🎙️ Image-to-Speech Generator


A lightweight Streamlit application that converts an image into a caption, transforms the caption into a short story, and finally generates speech from the story.
Powered by BLIP, LLaMA 3 (Ollama), and gTTS.

🚀 Features
Image Captioning using Salesforce/blip-image-captioning-base
Short Story Generation (≤50 words) using LLaMA 3 through ChatOllama
Text-to-Speech Conversion using gTTS
Simple Streamlit Interface to upload images and play generated audio


🛠️ Tech Stack
Step	Technology
Image → Caption	HuggingFace BLIP
Caption → Story	LLaMA 3 (Ollama) + LangChain
Story → Speech	gTTS (Google Text-to-Speech)
UI	Streamlit


📦 Installation
1. Clone the Repository
git clone https://github.com/Simarjeet06/Image-To-Speech
cd Image-To-Speech
2. Install Dependencies
pip install -r requirements.txt
3. Set Environment Variables
Create a .env file in the project root:
HUGGINGFACE_API_TOKEN=<your_huggingface_token>
4. Install & Start Ollama
Download: https://ollama.ai
Then pull the required model:
ollama pull llama3
Keep Ollama running in the background.
5. Run the App
streamlit run app.py



📌 How to Use
Upload an image (jpg, jpeg, png)
The app automatically:
Generates a caption
Produces a short story
Converts it to audio
Play the generated speech directly from the UI



📂 Project Structure

Image-To-Speech/
│── app.py
│── requirements.txt
│── .env
│── utils/
│   └── custom.py
