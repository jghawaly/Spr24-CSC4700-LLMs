from openai import OpenAI
import base64
import os
from dotenv import load_dotenv


# load environmental variables
load_dotenv('../../.env')

# establish client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def encode_image(image_path):
    """Encodes an image to base64 format for OpenAI API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def classify_objects_in_image(image_path):
    """Sends an image to OpenAI's GPT-4o-mini and gets classified objects."""

    # Convert image to base64
    base64_image = encode_image(image_path)

    # Call OpenAI GPT-4o-mini with vision capabilities
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI model trained to identify objects in images."},
            {"role": "user", "content": [
                {"type": "text", "text": "Identify as many objects in this image as possible."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": 'high'}}
            ]}
        ],
        max_tokens=500
    )

    # Extract response
    return response.choices[0].message.content


# Example usage
image_path = "/Users/jghawaly/Documents/nola_stock_photo.jpg"
classified_objects = classify_objects_in_image(image_path)
print("Classified Objects:", classified_objects)
