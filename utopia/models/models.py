import base64
import google.generativeai as genai
import math
import numpy as np
import os
import sys
from PIL import Image
from io import BytesIO
from openai import AzureOpenAI, OpenAI, BadRequestError
from typing import List, Optional, Union, Dict, Tuple
import json
import traceback


class LLM:
    def __init__(self, model_name):
        self.model_name = model_name
        
    def generate(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
    

class GPT(LLM):

    def __init__(self, model_name):
        super().__init__(model_name)
        if os.getenv("AZURE_OPENAI_ENDPOINT", None) is not None:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if azure_endpoint is None:
                raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")
            self.client = AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                                      api_version=os.getenv('OPENAI_API_VERSION'),
                                      azure_endpoint=azure_endpoint,
                                      azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT")
                                      )

        elif os.getenv("OPENAI_API_KEY", None) is not None:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError("No API key found")

    def generate(self, prompt: Optional[str] = None, image: Optional[List[Union[str, Image.Image]]] = None, temperature: float = 0.7, system_prompt: Optional[str] = "This is a simulation of an academic ecosystem, where researchers choose research directions and submit papers, reviewers conduct peer reviews of papers, and funding agencies allocate fundings.", response_format = None, message_history: Optional[List[Dict]] = []) -> Tuple[Dict, List[Dict]]:
        # System instructions for the GPT model

        try:
            # Handle string prompt with image formatting similar to process_sample_proprietary_models
            content = [
                {
                    "type": "text",
                    "text": prompt,
                }
            ]

            # Add images if provided
            if image is not None:
                if isinstance(image, Image.Image):
                    image = [image]
                
                if isinstance(image, list):
                    for img in image:
                        if isinstance(img, str):
                            encoded_image = load_base64_image(img)
                        else:
                            # Convert PIL Image to base64
                            img_buffer = BytesIO()

                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            img.save(img_buffer, format='JPEG')
                            encoded_image = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                        })

            
            messages = []
            if message_history != []:
                new_message = [{
                    "role": "user",
                    "content": content
                }]
                messages = message_history + new_message
                
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ]
                
            assert len(messages) > 0, "Message history is empty"

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"} # response_format
            )

            response_content = response.choices[0].message.content
            
            if response_content is None:
                raise ValueError("Empty response from model")
            
            response_json = json.loads(response_content.strip())
            messages.append({
                "role": "assistant",
                "content": [{
                    "type": "text", 
                    "text": response_content
                }]
            })


            return response_json, messages

        except BadRequestError as e:
            print(f"Error generating content with GPT-4o")
            traceback.print_exc()
            return None


class Gemini(LLM):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name)
        self.model_name = model_name
        self.max_tokens = kwargs.get('max_tokens', 1024)
        self.temperature = kwargs.get('temperature', 0.0)

        # Check if Google API key is set
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini models")

        # Configure the API
        genai.configure(api_key=api_key)

        # Initialize the Gemini model
        self.client = genai.GenerativeModel(model_name)
        self.config = genai.types.GenerationConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature
        )

    def generate(self, image: Union[str, Image.Image, List[Union[str, Image.Image]]] = None, prompt: str = None,
                 temperature: float = 0.0, top_p: float = 0.7):
        # Update config if temperature is provided
        if temperature != self.temperature:
            self.config = genai.types.GenerationConfig(
                max_output_tokens=self.max_tokens,
                temperature=temperature
            )

        content_parts = []
        # Add text prompt first

        # Add images if provided
        if image is not None:
            if isinstance(image, list):
                for img in image:
                    if isinstance(img, str):
                        img = Image.open(img).convert('RGB')
                    content_parts.append(img)
            else:
                if isinstance(image, str):
                    image = Image.open(image).convert('RGB')
                content_parts.append(image)

        if prompt:
            content_parts.append(prompt)

        try:
            response = self.client.generate_content(content_parts, generation_config=self.config)
            response.resolve()
            return response.text.strip()
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            return f"Error: {str(e)}"

    def complete(self, prompt: str):
        """
        Generate text-only response (for compatibility with other models).
        
        Args:
            prompt: Text prompt
            
        Returns:
            Generated text response
        """
        return self.generate(prompt=prompt)

    def embed_query(self, text: str):
        """
        Generate embeddings for text (for compatibility with other models).
        Note: This is a placeholder implementation since Gemini doesn't have a direct embedding API.
        
        Args:
            text: Text to embed
            
        Returns:
            Placeholder embedding (zeros vector)
        """
        # For now, return a placeholder embedding since Gemini doesn't have a direct embedding API
        # This will cause the similarity computation to return random results
        return np.zeros(768)  # Return a zero vector as placeholder


def load_base64_image(image_input: Union[str, Image.Image], format: str = "JPEG") -> str:
    """
    Encodes an image as a base64 string.

    This function can handle both file paths and PIL Image objects. For PIL Images,
    it converts them to RGB format if needed and saves them in the specified format.

    Args:
        image_input (Union[str, Image.Image]): Either a path to an image file or a PIL Image object.
        format (str): Image format for PIL Images (default: "JPEG"). Ignored for file paths.

    Returns:
        str: Base64-encoded string representation of the image.
    """
    if isinstance(image_input, str):
        # Handle file path - read directly as bytes
        with open(image_input, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    elif isinstance(image_input, Image.Image):
        # Handle PIL Image - convert to RGB if needed and encode
        img = image_input
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        buffer = BytesIO()
        img.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}. Expected str (file path) or PIL.Image")

