import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import sounddevice as sd  # For audio playback
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# === LangChain Settings ===
# LangChain LLM (Large Language Model)
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.7, groq_api_key="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")

# Prompt Template for LangChain
prompt_template = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(
        "You are a virtual human resource assistant. Ask questions to the candidate:\n{context}"
    )
])

# === Parler-TTS Settings ===
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load Parler-TTS Model
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

def text_to_speech(prompt, description=("Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise.")):
    """
    Convert the given text to speech and play the audio.
    """
    input_ids = tokenizer(description, return_tensors="pt", truncation=True).input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(device)

    # Generate audio
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids, temperature=0.3)
    audio_arr = generation.cpu().numpy().squeeze()

    # Play audio
    sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)
    sd.play(audio_arr, samplerate=model.config.sampling_rate)
    sd.wait()  # Wait for playback to finish

def ask_question(context):
    """
    Generate a question using LangChain LLM and convert it to speech.
    """
    # Generate a question from LangChain
    prompt = prompt_template.format_prompt(context=context)
    response = llm(prompt.to_messages())
    question = response.content.strip()
    print(f"Generated question: {question}")

    # Convert question to speech
    text_to_speech(question)
    return question

# === Main Function ===
def main():
    print("Welcome to the virtual HR assistant!")
    context = "Please introduce yourself and describe your professional background."

    while True:
        # Ask a question
        question = ask_question(context)

        # Get user response
        user_response = input("Your answer (type 'exit' to quit): ")
        if user_response.lower() in ["exit", "quit"]:
            print("Thank you for using the HR assistant. Goodbye!")
            break

        # Set the next context (for demonstration purposes)
        context = "What are your key skills and accomplishments?"

if __name__ == "__main__":
    main()