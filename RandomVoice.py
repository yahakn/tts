import time  # Measure execution time
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import sounddevice as sd  # For audio playback
from concurrent.futures import ThreadPoolExecutor
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# === LangChain Settings ===
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.3, groq_api_key="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")

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

# === Parler-TTS Settings ===
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load Parler-TTS model and tokenizer
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

def process_chunk(chunk, description):
    """
    Process a single chunk of text and convert it to speech with attention masks.
    """
    try:
        # Tokenize the chunk and description
        tokens = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=128,  # Daha büyük bir maksimum uzunluk
            padding=True
        ).to(device)

        description_tokens = tokenizer(
            description,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(device)

        # Attention mask oluşturma
        prompt_attention_mask = (tokens["input_ids"] != tokenizer.pad_token_id).long()
        description_attention_mask = (description_tokens["input_ids"] != tokenizer.pad_token_id).long()

        # Generate audio with the model
        generation = model.generate(
            input_ids=description_tokens["input_ids"],
            attention_mask=description_attention_mask,
            prompt_input_ids=tokens["input_ids"],
            prompt_attention_mask=prompt_attention_mask,
            temperature=0.3,  # Daha iyi ses kalitesi için sıcaklık artırıldı
            max_length=250  # Maksimum uzunluk artırıldı
        )

        # Convert the generated output to a tensor
        audio_chunk = torch.tensor(generation.cpu().numpy().squeeze())
        return audio_chunk

    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory. Reducing chunk size or model parameters may help.")
        torch.cuda.empty_cache()
        return None

    except Exception as e:
        print(f"Error in processing chunk: {e}")
        return None
    
def text_to_speech(
    prompt, 
    description="Jordan speaks slightly faster than usual, but in a monotone voice."
):
    """
    Convert the text to speech in parallel chunks and play the combined audio.
    """
    start_time = time.time()

    # Split prompt into smaller chunks
    max_chunk_length = 10  # Chunk uzunluğu azaltıldı
    words = prompt.split()
    chunks = [' '.join(words[i:i + max_chunk_length]) for i in range(0, len(words), max_chunk_length)]

    print(f"Split text into {len(chunks)} chunks for parallel processing.")

    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda chunk: process_chunk(chunk, description), chunks))

    # Filter out failed chunks
    audio_chunks = [chunk for chunk in results if chunk is not None]

    if not audio_chunks:
        print("No audio chunks were generated. Please check the input.")
        return

    # Combine audio chunks
    final_audio = torch.cat(audio_chunks)
    print(f"Final audio length: {len(final_audio)} samples")

    # Play and save the combined audio
    sd.play(final_audio.numpy(), samplerate=model.config.sampling_rate)
    sd.wait()
    sf.write("parler_tts_out.wav", final_audio.numpy(), model.config.sampling_rate)

    end_time = time.time()
    print(f"Audio streaming completed and saved to parler_tts_out.wav.")
    print(f"Text-to-Speech execution time: {end_time - start_time:.2f} seconds")

def generate_question(context):
    """
    Use LangChain (Groq) to generate a question based on the context.
    """
    try:
        # Generate a question
        prompt = prompt_template.format_prompt(context=context)
        response = llm(prompt.to_messages())
        question = response.content.strip()
        print(f"Generated Question: {question}")
        return question
    except Exception as e:
        print(f"Error in generating question: {e}")
        return "Could you elaborate on your experience?"

def hr_assistant_main():
    """
    Main function to run the HR Assistant.
    """
    print("Welcome to the Virtual HR Assistant!")
    context = "Please introduce yourself and describe your professional background."

    while True:
        # Generate a question
        question = generate_question(context)

        # Convert the question to speech
        text_to_speech(question)

        # Get user response (mock interaction)
        user_response = input("Your answer (type 'exit' to quit): ")
        if user_response.lower() in ["exit", "quit"]:
            print("Thank you for using the HR Assistant. Goodbye!")
            break

        # Update the context for the next question
        context = f"The candidate responded: {user_response}. Ask a relevant follow-up question."

if __name__ == "__main__":
    hr_assistant_main()