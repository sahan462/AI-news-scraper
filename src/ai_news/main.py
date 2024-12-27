#!/usr/bin/env python
import os
import sys
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Force litellm to use Cohere before any other imports
os.environ["OPENAI_API_KEY"] = "not-needed"  # Dummy value to prevent the error
os.environ["COHERE_API_KEY"] = os.getenv('COHERE_API_KEY')
os.environ["LLM_PROVIDER"] = "cohere"

# Now import the rest
import warnings
from datetime import datetime
from ai_news.crew import AiNews
from crewai import Agent, LLM

# Check for Cohere API key
if not os.getenv('COHERE_API_KEY'):
    print("Error: COHERE_API_KEY is missing. Please check your .env file.", file=sys.stderr)
    sys.exit(1)

# Configure Cohere LLM
llm = LLM(
    provider="cohere",
    model="command-r-plus-08-2024",
    temperature=0.7,
    api_key=os.getenv('COHERE_API_KEY'),
    context_window=2048,  # Explicitly set context window
    max_tokens=500,       # Explicitly set max tokens
)

# Override the default LLM settings
Agent.default_llm = llm

print("Cohere LLM configured successfully.")

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'AI LLMs',
        'date': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    }
    AiNews().crew().kickoff(inputs=inputs)

if __name__ == "__main__":
    run()