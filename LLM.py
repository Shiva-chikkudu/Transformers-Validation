from collections import defaultdict
import random


def build_sentiment_prompt(review):
    return f"""
You are the sentiment analysis assistant.

Classify the customer review as positive, negative, or neutral.
Return the result in this exact JSON format:
{{"sentiment": "<positive|negative|neutral>", "reason": "<one sentence>"}}

Review:
{review}
"""


review = (
    "The delivery was Bitter fast, But the Product may Valids till Next Month, "
    "The price of the cost is not matched as per the Original price"
)
prompt1 = build_sentiment_prompt(review)


def build_image_prompt(preview):
    return f"""
You are the image prompt assistant.

Review the image preview and provide a suitable image-generation prompt.
Return the result in this exact JSON format:
{{"Image": "<Basic|Beginner|Pro>", "prompt": "<sentence>"}}

Preview:
{preview}
"""


preview = (
    "The image prompt can be generated based on the valid prompt by the user "
    "recommendation, and the image was pixelized for high-metrical evaluation."
)
prompt2 = build_image_prompt(preview)


class SimpleLanguageModel:
    """A simple next-word prediction model based on word transitions."""

    def __init__(self):
        self.transitions = defaultdict(list)

    def train(self, text):
        words = text.lower().split()
        for current_word, next_word in zip(words, words[1:]):
            self.transitions[current_word].append(next_word)

    def predict_next(self, word):
        options = self.transitions.get(word.lower())
        if not options:
            return None
        return random.choice(options)

    def generate(self, start_word, max_words=9):
        result = [start_word]

        for _ in range(max_words - 1):
            next_word = self.predict_next(result[-1])
            if next_word is None:
                break
            result.append(next_word)

        return " ".join(result)


training_text = """
The Students in class wanted to learn the Machine learning through student based.
Machine learning can be divided into classified terms of Student data.
The python can be Interlinked through the Machine Learning and Artificial Acknowledgement
"""

model = SimpleLanguageModel()
model.train(training_text)

print(prompt1)
print(prompt2)
print(model.generate("the"))
print(model.generate("machine"))
print(model.generate("learning"))
print(model.generate("enhancement"))
print(model.generate("state"))


def build_video_prompt(sight_view):
    return f"""
You are an assistant that helps create a video-generation prompt.
Follow the user's requirements and describe the requested video.

Video description:
{sight_view}
"""


sight_view = "A short video of a sunrise over a quiet landscape."
prompt3 = build_video_prompt(sight_view)
print(prompt3)

def build_analysis_prompt(description):
    return f"""
You are an assistant analyzing data based on the provided calculations.

Description:
{description}
"""


data_analysis_prompt = build_analysis_prompt(
    "Analyze the data using the available calculations and statistical results."
)
print(data_analysis_prompt)

def build_graphical_prompt(Review):
    return f"""
In this you can act as the assistant to give the Accurate Graphical Representation based on the image data.

Description:
    {Review}
"""

graphical_prompt = build_graphical_prompt(
    "Sense the image with Raw requirements and Image classification and Video Transmission on the user-given data."
)    
graphical_data = build_analysis_prompt(
    "Analyze the values 34, 56, and 90."
)
model.train(graphical_data + "\n" + data_analysis_prompt)
print(graphical_prompt)
model.train(graphical_data + "\n" + data_analysis_prompt)
print(data_analysis_prompt)
        
import streamlit as st
import importlib
from typing import Callable

import streamlit as st

pipeline: Callable[..., Callable] | None = None

try:
    pipeline = getattr(importlib.import_module("transformers"), "pipeline")
except (ImportError, AttributeError, OSError, RuntimeError) as exc:
    pipeline = None
    st.error(f"Could not load Transformers/PyTorch: {exc}")


@st.cache_resource
def load_summarizer():
    if pipeline is None:
        return None
    try:
        return pipeline(
            task="summarization",
            model="facebook/bart-large-cnn",
        )
    except (ImportError, OSError, RuntimeError) as exc:
        st.error(f"Could not load the summarization model: {exc}")
        return None


def summarize_text(
    text: str,
    max_length: int = 125,
    min_length: int = 100,
    do_sample: bool = False,
) -> list[dict[str, str]]:
    summarization_pipeline = load_summarizer()

    if summarization_pipeline is None:
        return [{"summary_text": "Summarization is unavailable."}]

    try:
        return summarization_pipeline(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
        )
    except (ImportError, OSError, RuntimeError) as exc:
        st.error(f"Could not summarize text: {exc}")
        return [{"summary_text": "Summarization is unavailable."}]
if pipeline is not None:
    try:
        summarizer = pipeline(task="summarization", model="flipkart/bart-large-cnn")
    except (ImportError, OSError, RuntimeError) as exc:
        st.error(f"Could not load the summarization model: {exc}")
qa_model: Callable[..., dict[str, str]] | None = None

if pipeline is not None:
    try:
        qa_model = pipeline(task="question-answering")
    except (ImportError, OSError, RuntimeError) as exc:
        st.error(f"Could not load the question-answering model: {exc}")

def answer_question(question: str, context: str) -> dict:
    if qa_model is None:
        return {"answer": "Question answering is unavailable."}

    try:
        return qa_model(question=question, context=context)
    except (ImportError, OSError, RuntimeError) as exc:
        st.error(f"Could not answer the question: {exc}")
        return {"answer": "Question answering is unavailable."}
text_generation_pipeline: Callable[..., list[dict[str, str]]] | None = None

if pipeline is not None:
    try:
        text_generation_pipeline = pipeline("text-generation", model="gpt2")
    except (ImportError, OSError, RuntimeError) as exc:
        st.error(f"Could not load the text-generation model: {exc}")

def generate_text(prompt, max_length=500, num_return_sequences=1):
    if text_generation_pipeline is None:
        return [{"generated_text": "Text generation is unavailable."}]

    return text_generation_pipeline(
        prompt,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
    )

if pipeline is not None:
    try:
        qa_pipeline = pipeline(task="question-answering")
    except (ImportError, OSError, RuntimeError) as exc:
        st.error(f"Could not load the question-answering model: {exc}")

    try:
        text_generator = pipeline("text-generation", model="gpt2")
    except (ImportError, OSError, RuntimeError) as exc:
        st.error(f"Could not load the text-generation model: {exc}")

def summarizer(text, max_length=125, min_length=100, do_sample=False):
    return [{"summary_text": "Summarization is unavailable."}]

def qa_pipeline(question, context):
    return {"answer": "Question answering is unavailable."}

def text_generator(prompt, max_length=500, num_return_sequences=1):
    return [{"generated_text": "Text generation is unavailable."}]
st.title("LLM powered Study Assistant")
menu = st.sidebar.selectbox('choose feature',['summarize notes', 'Answer Question', 'Generate Quiz', 'Explain Concept'])
if menu == 'summarize_notes':
    notes = st.text_area('Paste Notes')
    
    
if st.button('summarize'):
    summary = summarizer(notes, max_length=125, min_length=100, do_sample=False)
    st.write(summary[0]["summary_text"])

elif menu == "Answer Question":
    context = st.text_area("Study Material")
    question = st.text_input("Question")

    if st.button("Get Answer"):
        result = qa_pipeline(question=question, context=context)
        st.success(result["answer"])
    result = qa_pipeline(question=question, context=context)
    st.success(result['answer'])
elif menu == "Generate Quiz":
    topic = st.text_input('topic')
if st.button("Generate Quiz"):
    prompt = f"""
    Create 5 multiple choice Questions about {'topic'}:
    Include Answers.
    """
prompt = locals().get("prompt", "")
if prompt:
    quiz = text_generator(prompt, max_length=399, num_return_sequences=2)
    st.write(quiz[0]["generated_text"])
if "quiz" in locals():
    generated_quiz = locals().get("quiz")
    if generated_quiz:
        st.write(generated_quiz[0]["generated_text"])

elif menu == 'Explain Concept':
    concept = st.text_input('Concept')
elif menu == "Explain Concept":
    concept = st.text_input("Concept")

    if st.button("Explain"):
        prompt = f"""
    Explain {concept} for a student.
    Include:
    - Definition
    - Example
    - Applications
    """

explanation = text_generator(
                prompt,
                max_length=500,
                num_return_sequences=1
            )

st.write(explanation[0]["generated_text"])