import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.core_skills import TimeSkill
from IPython.display import display, Markdown
import asyncio
import json

kernel = sk.Kernel()

useAzureOpenAI = False

deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
kernel.add_text_completion_service("azureopenai", AzureChatCompletion(deployment, endpoint, api_key))

print("You made a kernel.")

from semantic_kernel.skill_definition import (
    sk_function,
    sk_function_context_parameter,
)

async def main():

    sk_prompt = """
    A 40-year old man who has just finished his shift at work and comes into the bar. They are in a bad mood.

    They are given an experience like:
    {{$input}}

    Summarize their possible reactions to this experience.
    """
    test_function = kernel.create_semantic_function(prompt_template=sk_prompt,
                                                        description="Simulates reaction to an experience.",
                                                        max_tokens=1000,
                                                        temperature=0.1,
                                                        top_p=0.5)
    sk_input="""
    A simple loyalty card that includes details such as the rewards for each level of loyalty, how to earn points, and how to redeem rewards is given to every person visiting the bar.
    """

    test_result = await kernel.run_async(test_function, input_str=sk_input) 

    print(str(test_result))

asyncio.run(main())