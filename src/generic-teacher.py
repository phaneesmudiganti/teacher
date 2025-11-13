from dotenv import load_dotenv
from openai import OpenAI
import json
import os
from pypdf import PdfReader
import gradio as gr
from gtts import gTTS
import tempfile
import re

load_dotenv(override=True)

# Utility: Clean text for audio
def clean_text_for_audio(text):
    # Remove markdown formatting
    text = re.sub(r'\*\*|\*|__|~~|`|–|-', '', text)

    # Remove transliterations in parentheses (e.g., (Grandmother))
    text = re.sub(r'\([^()\n]+\)', '', text)

    # Add a period after numbered items with Hindi word
    text = re.sub(r'(\d+\.\s*)([^\n]+)', r'\1\2.', text)

    # Add a period after English meaning lines
    text = re.sub(r'(English meaning:.*?)\n', r'\1. ', text)

    # Add a pause before any example or explanation line
    text = re.sub(r'\n\s*[\*\-•]\s*', r'. ', text)

    # Replace multiple spaces and newlines with single space
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.replace('\n', ' ').strip()

    return text

# Tool definitions (no PushOver)
record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"},
            "name": {"type": "string", "description": "The user's name, if they provided it"},
            "notes": {"type": "string", "description": "Any additional information about the conversation"}
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question that couldn't be answered"},
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
         {"type": "function", "function": record_unknown_question_json}]

def format_history(history):
    formatted = ""
    for msg in history:
        role = msg["role"].capitalize()
        content = msg["content"]
        formatted += f"**{role}:** {content}\n\n"
    return formatted

# Main Agent Class
class SubjectTeacherAgent:

    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.openai = OpenAI(api_key=self.google_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

    def load_pdf_content(self, subject, chapter_number):
        path = f"content/{subject.lower()}/chapter{chapter_number}.pdf"
        if not os.path.exists(path):
            return f"No content found for subject: {subject} Chapter {chapter_number}."
        reader = PdfReader(path)
        content = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                content += text
        return content

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            # No PushOver logic — just simulate tool response
            result = {"recorded": "ok"}
            results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
        return results

    def system_prompt(self, subject, chapter_number, chapter_content):
        return (
            f"You are an Autonomous Agent acting as a CBSE Grade 2 {subject} Teacher.\n"
            f"Your role is to help young students understand {subject} chapters from their curriculum.\n\n"
            f"You will be provided with the chapter number ({chapter_number}) and the full text of the chapter.\n\n"
            "Your tasks are:\n"
            "1. Comprehend the chapter content and explain it in simple, age-appropriate language suitable for a Grade 2 student.\n"
            "2. If the subject is a language (like Hindi), translate each line into English to help students understand the meaning more clearly.\n"
            "3. Identify and list difficult words or concepts from the chapter and provide their meanings or explanations.\n\n"
            "You must respond to any student question that relates to the chapter, even if the question is casual, vague, or phrased in a friendly way.\n"
            "If the student greets you or asks a general question, respond warmly and then guide them toward learning from the chapter.\n"
            "If the student asks about difficult words, translations, or explanations, use the chapter content to answer clearly.\n"
            "If the student asks something unrelated to the chapter or subject, use the record_unknown_question tool.\n"
            "If the student shares their email or wants to stay in touch, use the record_user_details tool.\n\n"
            f"## Chapter Content:\n{chapter_content}\n\n"
            f"With this context, please chat with the user, always staying in character as a CBSE Grade 2 {subject} Teacher."
        )

    def chat(self, subject, chapter_number, message, history=None):
        if history is None:
            history = []

        chapter_content = self.load_pdf_content(subject, chapter_number)
        prompt = self.system_prompt(subject, chapter_number, chapter_content)

        messages = [{"role": "system", "content": prompt}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gemini-2.5-flash", messages=messages, tools=tools)
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True

        response_text = response.choices[0].message.content

        # Add assistant response to history
        messages.append({"role": "assistant", "content": response_text})

        # Generate TTS audio
        cleaned_text = clean_text_for_audio(response_text)
        tts = gTTS(text=cleaned_text, lang='hi' if subject.lower() == "hindi" else 'en')
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio.name)

        formatted_history = format_history(messages)
        return response_text, temp_audio.name, messages, formatted_history


# Gradio UI
if __name__ == "__main__":
    agent = SubjectTeacherAgent()
    gr.Interface(
        fn=agent.chat,
        inputs=[
            gr.Dropdown(label="Subject", choices=["Hindi", "Math", "Science"]),
            gr.Dropdown(label="Chapter Number", choices=["1", "2", "3", "4", "5", "6"]),
            gr.Textbox(label="Your message", placeholder="Type your question here..."),
            gr.State()
        ],
        outputs=[
            gr.TextArea(label="Response", lines=10),
            gr.Audio(label="Voice Response", type="filepath"),
            gr.State(),
            gr.Markdown(label="Chat History")
        ],
        title="CBSE Grade 2 Subject Teacher",
        description="Select a subject and ask questions about the chapter!"
    ).launch(inbrowser=True)