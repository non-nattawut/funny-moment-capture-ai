from langchain_openai import ChatOpenAI
from src.config import Config

class LLMFactory:
    @staticmethod
    def get_llm():
        """Returns the appropriate ChatOpenAI instance based on the LLM_PROVIDER setting."""

        if Config.LLM_PROVIDER == "nvidia":
            print(f"--- Initializing NVIDIA NIM LLM ({Config.NVIDIA_MODEL_NAME}) ---")
            return ChatOpenAI(
                base_url=Config.NVIDIA_BASE_URL,
                api_key=Config.NVIDIA_API_KEY,
                model=Config.NVIDIA_MODEL_NAME,
                temperature=0.7
            )

        # Default to LM Studio
        print(f"--- Initializing LM Studio LLM ({Config.LM_STUDIO_MODEL_NAME}) ---")
        return ChatOpenAI(
            base_url=Config.LM_STUDIO_BASE_URL,
            api_key=Config.LM_STUDIO_API_KEY,
            model=Config.LM_STUDIO_MODEL_NAME,
            temperature=0.7
        )
