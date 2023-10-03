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

    from semantic_kernel.planning.sequential_planner.sequential_planner_config import SequentialPlannerConfig
    from semantic_kernel.planning import SequentialPlanner    

    plugins_directory = "./plugins-sk"
    writer_plugin = kernel.import_semantic_skill_from_directory(plugins_directory, "LiterateFriend")

    planner = SequentialPlanner(kernel, SequentialPlannerConfig(excluded_skills=["this"]))

    ask = """
    Tomorrow is Valentine's day. I need to come up with a poem. Translate the poem to French.
    """

    plan = await planner.create_plan_async(goal=ask)
    planner = SequentialPlanner(kernel, SequentialPlannerConfig(excluded_skills=["this"]))
    result = await plan.invoke_async()

    for index, step in enumerate(plan._steps):
        print(f"✅ Step {index+1} used function `{step._function.name}`")

    trace_resultp = True

    if trace_resultp:
        print("Longform trace:\n")
        for index, step in enumerate(plan._steps):
            print("Step:", index)
            print("Description:",step.description)
            print("Function:", step.skill_name + "." + step._function.name)
            #print("Input vars:", step._parameters._variables)
            #print("Output vars:", step._outputs)
            if len(step._outputs) > 0:
                print( "  Output:\n", str.replace(result[step._outputs[0]],"\n", "\n  "))

    display(Markdown(f"## ✨ Generated result from the ask: {ask}\n\n---\n" + str(result)))


asyncio.run(main())