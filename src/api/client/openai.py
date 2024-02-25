# stdlib
import argparse
import base64
import os

# third-party
from dotenv import load_dotenv
from openai import OpenAI


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class OpenAIClient(object):

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4-vision-preview"

    def image_to_text(self, prompt, img_path, **kwargs):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_path}},
                ],
            }
        ]

        temperature = kwargs.get("temperature", 0)
        max_tokens = kwargs.get("max_tokens", 2024)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
        )

        output = response.choices[0]
        if output.finish_reason != "stop":
            print("Model didn't stop natrually")
            print("Finish reason:", output.finish_reason)

        return output.message.content

    def vision_image_url(self, prompt, image_url, **kwargs):
        return self.image_to_text(prompt, image_url)

    def vision_image_file(self, prompt, image_file, **kwargs):
        base64_image = encode_image(image_file)
        image_url = f"data:image/jpeg;base64,{base64_image}"
        return self.image_to_text(prompt, image_url)

    def vision_image_bytes(self, prompt, image_bytes, **kwargs):
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{base64_image}"
        return self.image_to_text(prompt, image_url)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-u", "--image_url", help="Image URL")
    argparser.add_argument("-f", "--image_file", help="Image File")

    prompt = "What’s in this image?"
    args = argparser.parse_args()
    oac = OpenAIClient()
    if args.image_url:
        print("Processing image URL", args.image_url)
        response = oac.vision_image_url(prompt, args.image_url)
        print(response)
    elif args.image_file:
        print("Processing image file", args.image_file)
        response = oac.vision_image_file(prompt, args.image_file)
        print(response)
    else:
        argparser.print_help()


if __name__ == "__main__":
    main()
