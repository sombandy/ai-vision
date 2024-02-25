# first-party
from io import StringIO
import json
from src.api.client.openai import OpenAIClient

# third-party
import pandas as pd
import streamlit as st


def json_to_df(json_str):
    json_str = json_str.strip()
    try:
        json_obj = json.loads(json_str)
        df = pd.DataFrame(json_obj['companies'])
        return df
    except (json.JSONDecodeError, KeyError):
        lines = json_str.split('\n')

        print("Incorrect JSON format")
        print("Trying removing the last line", lines[-1])

        if lines[-2].strip().endswith('},'):
            lines[-2] = lines[-2].strip()[:-1]
            fixed_json = '\n'.join(lines[:-1]) + '\n]}'
            try:
                json_obj = json.loads(fixed_json)
                df = pd.DataFrame(json_obj['companies'])
                return df
            except (json.JSONDecodeError, KeyError):
                print("Failed to load JSON")
                print("Original JSON:", json_str)

                return pd.DataFrame()


st.set_page_config(page_title="Webd2Lead", page_icon="🤖")
st.header("Extract company information from an image", divider='rainbow')


image_file = st.file_uploader(
    "Upload an image with company names or logos...", type=["jpg", "jpeg", "png", "webp"]
)

prompt_file = "src/prompt/extractor.txt"
with open(prompt_file, "r") as file:
    prompt = file.read()

oac = OpenAIClient()
if image_file is not None:
    bytes_data = image_file.getvalue()
    response = oac.vision_image_bytes(prompt, bytes_data)

    df = json_to_df(response)
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.write(response)