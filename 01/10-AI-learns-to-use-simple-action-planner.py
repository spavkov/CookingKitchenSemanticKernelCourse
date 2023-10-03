import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAITextEmbedding
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding

from semantic_kernel.core_skills import TimeSkill
from IPython.display import display, Markdown
import asyncio
import json

kernel = sk.Kernel()

useAzureOpenAI = False

deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
print(endpoint)
kernel.add_text_completion_service("azureopenai", AzureChatCompletion(deployment, endpoint, api_key))

print("You made a kernel.")

from semantic_kernel.skill_definition import (
    sk_function,
    sk_function_context_parameter,
)

async def main():

    deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
    kernel.add_text_completion_service("azureopenaicompletion", AzureChatCompletion(deployment, endpoint, api_key))
    kernel.add_text_embedding_generation_service("azureopenaiembedding", AzureTextEmbedding("text-embedding-ada-002", endpoint, api_key))

    print("I did it boss!")

    from semantic_kernel.planning import ActionPlanner

    planner = ActionPlanner(kernel)

    from semantic_kernel.core_skills import FileIOSkill, MathSkill, TextSkill, TimeSkill
    kernel.import_skill(MathSkill(), "math")
    kernel.import_skill(FileIOSkill(), "fileIO")
    kernel.import_skill(TimeSkill(), "time")
    kernel.import_skill(TextSkill(), "text")

    print("Adding the tools for the kernel to do math, to read/write files, to tell the time, and to play with text.")

    ask = "What is the sum of 110 and 990?"

    print(f"🧲 Finding the most similar function available to get that done...")
    plan = await planner.create_plan_async(goal=ask)
    print(f"🧲 The best single function to use for task `{ask}`is `{plan._skill_name}.{plan._function.name}`")
    timeAsk = "what day is today?"
    plan = await planner.create_plan_async(goal=timeAsk)
    print(f"🧲 The best single function to use for task `{timeAsk}`is `{plan._skill_name}.{plan._function.name}`")


asyncio.run(main())