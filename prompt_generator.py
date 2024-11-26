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

def save_to_file(filename, content):
    """
    Save the generated question to a file.
    """
    try:
        with open(filename, "w") as file:
            file.write(content)
        print(f"Question saved to {filename}.")
    except Exception as e:
        print(f"Error in saving to file: {e}")

def hr_assistant_main():
    """
    Main function to run the HR Assistant.
    """
    print("Welcome to the Virtual HR Assistant!")
    context = "Please introduce yourself and describe your professional background."

    while True:
        # Generate a question
        question = generate_question(context)

        # Save the question to promp.txt
        save_to_file("promp.txt", question)

        # Get user response (mock interaction)
        user_response = input("Your answer (type 'exit' to quit): ")
        if user_response.lower() in ["exit", "quit"]:
            print("Thank you for using the HR Assistant. Goodbye!")
            break

        # Update the context for the next question
        context = f"The candidate responded: {user_response}. Ask a relevant follow-up question."

if __name__ == "__main__":
    hr_assistant_main()