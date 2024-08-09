#!/usr/bin/env python
"""
High level docs:
https://github.com/google-gemini/generative-ai-python/tree/main

Install the Google AI Python SDK
$ pip install google-generativeai

See the getting started guide for more information:
https://ai.google.dev/gemini-api/docs/get-started/python
"""

import os
import IPython

import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)

def upload_to_gemini(path: str, mime_type: str=None):
  """Uploads the given file to Gemini.

  See https://ai.google.dev/gemini-api/docs/prompting_with_media
  """
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

files = [
  upload_to_gemini("what_shape_comes_next.jpg", mime_type="image/jpeg"),
]

chat_session = model.start_chat(
  history=[{
    "role": "user",
    "parts": [
      files[0]
      ]
    }])

message: str = (
  "Look at this sequence of three shapes. What shape should come "
  "as the fourth shape? Explain your reasoning with detailed descriptions "
  "of the first shapes.")

IPython.embed()
response = chat_session.send_message(message)
print(response)

