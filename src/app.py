from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
from gtts import gTTS
import tempfile
import re

load_dotenv(override=True)


def clean_text_for_audio(text):
    # Remove markdown symbols and extra punctuation
    text = re.sub(r'\*\*|__|~~|`', '', text)  # Remove markdown formatting
    text = re.sub(r'[\"\'\[\]\(\)]', '', text)  # Remove quotes and brackets
    text = re.sub(r'\s{2,}', ' ', text)  # Replace multiple spaces with one
    return text.strip()


def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}


record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
         {"type": "function", "function": record_unknown_question_json}]


class Me:

    def __init__(self):
        self.google_api_key=os.getenv("GOOGLE_API_KEY")
        self.openai = OpenAI(api_key=self.google_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        reader = PdfReader("content/hindi/bhsr101.pdf")
        self.hindiChapterOne = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.hindiChapterOne += text

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
        return results

    def system_prompt(self):
        system_prompt = (
            "You are an Autonomous Agent acting as a CBSE Grade 2 Hindi Teacher. "
            "Your role is to help young students understand Hindi chapters from their curriculum.\n"
            "You will be provided with the chapter number and the full text of the chapter in Hindi.\n"
            "Your tasks are:\n\n"
            "Comprehend the chapter content and explain it in simple, age-appropriate language suitable for a Grade 2 student.\n"
            "Translate each line of the chapter into English to help students understand the meaning more clearly.\n"
            "Identify and list difficult Hindi words from the chapter and provide their meanings in English.\n\n"
            "Keep your tone friendly, encouraging, and suitable for young learners. Use simple vocabulary and short sentences.\n"
            "Wait for the input in the following format:\n"
            "Chapter Number: <number>\n"
            "Once received, begin your explanation, translation, and word meanings.\n"
            "If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
            If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "
        )

        system_prompt += f"\n\n## Chatper One Content:\n{self.hindiChapterOne}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as a CBSE Grade 2 Hindi Teacher."
        return system_prompt

    def chat(self, message, history=None):
        if history is None:
            history = []

        messages = [{"role": "system", "content": self.system_prompt()}] + history + [
            {"role": "user", "content": message}]
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

        # Generate TTS audio
        cleaned_text = clean_text_for_audio(response_text)
        tts = gTTS(text=cleaned_text, lang='en')
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio.name)

        return response_text, temp_audio.name


if __name__ == "__main__":
    me = Me()
    # gr.ChatInterface(me.chat, type="messages").launch()

    gr.Interface(
        fn=me.chat,
        inputs=gr.TextArea(label="Your message", lines=5, placeholder="Type your question here..."),
        outputs=[
            gr.TextArea(label="Response", lines=10),
            gr.Audio(label="Voice Response", type="filepath")
        ],
        title="CBSE Grade 2 Hindi Teacher",
        description="Ask questions about Hindi chapters and hear the answers!"
    ).launch(inbrowser=True)


