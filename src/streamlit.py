# first-party
from io import StringIO
import json
from src.api.client.openai import OpenAIClient

# third-party
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Vision", page_icon="🤖")
st.header("Extract company information from an image", divider="rainbow")


image_file = st.file_uploader(
    "Upload an image with company names or logos...",
    type=["jpg", "jpeg", "png", "webp"],
)

prompt_file = "src/prompt/extractor.txt"
with open(prompt_file, "r") as file:
    prompt = file.read()

oac = OpenAIClient()
if image_file is not None:
    st.image(image_file, use_column_width=True, caption="Uploaded Image")
    bytes_data = image_file.getvalue()
    response = oac.vision_image_bytes(prompt, bytes_data)

    companies = []
    with st.empty():
        for obj in response:
            companies.append(obj)
            st.write(json.dumps(obj, indent=2))
        st.write("Found %d companies" % len(companies))

    df = pd.DataFrame(companies)
    st.dataframe(df, use_container_width=True)
