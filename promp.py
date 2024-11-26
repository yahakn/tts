import time  # For measuring execution time
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import sounddevice as sd  # For audio playback
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

torch.cuda.empty_cache()

# === LangChain Model Settings ===
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.3, groq_api_key="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")

# === Device Settings ===
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# === Model and Tokenizer Loading ===
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

# Prompt Template (for generating HR questions)
prompt_template = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("""
        You are a virtual human resource assistant conducting a job interview. 
        Your goal is to ask short, professional, and engaging questions.

        Guidelines:
        1. Ask one question at a time.
        2. Use complete sentences and avoid jargon.
        3. Stay focused on the candidate's experience, skills, and goals.
        4. Ensure clarity and professionalism in tone.

        Start with: "Hello, could you please introduce yourself?"
    """)
])

def generate_question(context):
    """
    Use LangChain (Groq) to generate a question based on the context.
    """
    if not context:
        raise ValueError("Context cannot be empty. Please provide a valid context for question generation.")
    
    try:
        start_time = time.time()  # Start timer for question generation
        # Generate a question
        prompt = prompt_template.format_prompt(context=context)
        response = llm(prompt.to_messages())
        question = response.content.strip()
        end_time = time.time()  # End timer
        print(f"Generated Question: {question}")
        print(f"Question generation time: {end_time - start_time:.2f} seconds")
        return question
    except Exception as e:
        print(f"Error in generating question: {e}")
        return "Could you please elaborate on your experience?"

def tts_process(prompt):
    """
    Text-to-Speech process for converting prompt to audio.
    """
    if not prompt:
        raise ValueError("Prompt cannot be empty. Please provide a valid prompt for TTS.")

    try:
        description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."

        start_time = time.time()  # Start timer for TTS
        # Tokenizer with input IDs
        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Generate audio with the model
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids, temperature=0.7)

        # Save generated audio
        audio_arr = generation.cpu().numpy().squeeze()
        
        # Play and save the audio
        sd.play(audio_arr, samplerate=model.config.sampling_rate)
        sd.wait()
        sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)
        end_time = time.time()  # End timer
        print(f"Audio saved as 'parler_tts_out.wav'.")
        print(f"TTS process time: {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error in TTS process: {e}")
        exit(1)

def main():
    """
    Main function to orchestrate the execution.
    """
    # Initial context for the conversation
    context = "Please introduce yourself and describe your professional background. Initate the ınterview."

    while True:
        try:
            # Generate a question
            question = generate_question(context)

            # Convert the question to speech
            tts_process(question)

            # Mock interaction: Get user response
            user_response = input("Your answer (type 'exit' to quit): ")
            if user_response.lower() in ["exit", "quit"]:
                print("Thank you for using the HR Assistant. Goodbye!")
                break

            # Update the context for the next question
            context = f"The candidate responded: {user_response}. Ask a follow-up question related to their professional experience or career goals."
        except ValueError as ve:
            print(f"Input error: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
