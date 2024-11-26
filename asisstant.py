import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# === LangChain Ayarları ===
# OpenAI GPT modelini kullanarak bir ChatOpenAI nesnesi oluşturuyoruz
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.7, groq_api_key="key_here")

# Prompt Template (Kullanıcıyla etkileşim için)
prompt_template = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("""
       You are a virtual human resource assistant conducting a job interview with a candidate. Your primary objective is to create an interactive and engaging conversation by asking short and concise questions. Keep your tone professional, yet friendly and conversational.

Here are your rules:

1. **Ask short questions**: Each question should be no more than one or two sentences. Avoid long and complex questions.
2. **Follow up interactively**: Use the candidate's previous response to create a natural flow in the conversation.
3. **Stay on topic**: Focus on gathering key information about the candidate's professional background, skills, and goals.
4. **Limit responses**: Ensure your own responses are brief and to the point, guiding the conversation forward.
5. **Avoid repetition**: Do not ask the same question twice.
6. **Ask open-ended questions**: Encourage the candidate to elaborate without feeling overwhelmed.
7. **Provide context when necessary**: If a question requires explanation, add minimal context to ensure clarity.
8. **Do not use abbreviations or contractions**: Always use complete words and phrases to maintain clarity and professionalism.

Start the conversation by introducing yourself and asking a simple, open-ended question to initiate the dialogue."""
    )
])

# === Parler-TTS Ayarları ===
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Parler-TTS modelini yükle
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

def text_to_speech(prompt, description = "A calm and professional female speaker delivering clear and articulate speech in a natural and conversational tone, suitable for a human resources assistant. The voice exudes confidence and warmth, ensuring that every word is easily understood and approachable."):
    """
    Convert the text to speech and save it as a WAV file.
    """
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Ses üretimi
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    
    # Ses dosyasını yaz
    sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)
    print("Audio generated and saved as parler_tts_out.wav.")

def ask_questions(context):
    """
    Generate a question using LangChain and convert the response to speech with TTS.
    """
    # LangChain ile soru oluştur
    prompt = prompt_template.format_prompt(context=context)
    response = llm(prompt.to_messages())
    question = response.content.strip()

    print(f"Generated question: {question}")
    
    # Soruyu TTS ile sese dönüştür
    text_to_speech(question)
    return question

# === Ana Akış ===
def main():
    print("Welcome to the virtual HR assistant!")
    context = "Please introduce yourself and describe your professional background."
    question = ask_questions(context)

    # Kullanıcı yanıtını al
    user_response = input("Your answer: ")
    print(f"User Response: {user_response}")

    # Başka bir soru sorabilir veya işlemi devam ettirebilirsiniz
    next_context = "What are your key skills and accomplishments?"
    next_question = ask_questions(next_context)

if __name__ == "__main__":
    main()
