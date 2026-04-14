#!/usr/bin/env python3
"""
Test script for OpenAI-compatible LLM API
"""

import os
import json
from openai import OpenAI

def main():
    # Initialize the client with the provided base URL and API key
    client = OpenAI(
        base_url="https://albert.api.etalab.gouv.fr/v1",
        api_key=os.environ.get("ALBERT_API_KEY")
    )
    
    # Check if API key is set
    if not client.api_key:
        print("Error: ALBERT_API_KEY environment variable not set")
        return
    
    # Get available models to find a text-generation model
    try:
        models = client.models.list().data
        model = None
        for m in models:
            if m.type == "text-generation":
                model = m.id
                break
        
        if not model:
            print("Error: No text-generation model found")
            return
            
        print(f"Using model: {model}")
        
        # Send a test prompt
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Tu réponds en français, de façon concise."},
                {"role": "user", "content": "Explique ce qu’est une API compatible OpenAI en deux phrases."}
            ],
            stream=False
        )
        
        print("Response:")
        print(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()