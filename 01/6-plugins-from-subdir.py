import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.core_skills import TimeSkill
from IPython.display import display, Markdown
import asyncio

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

    my_context = kernel.create_new_context()

    strengths = [ "Unique garlic pizza recipe that wins top awards","Owner trained in Sicily at some of the best pizzerias","Strong local reputation","Prime location on university campus" ]
    weaknesses = [ "High staff turnover","Floods in the area damaged the seating areas that are in need of repair","Absence of popular calzones from menu","Negative reviews from younger demographic for lack of hip ingredients" ]


    pluginsDirectory = "./plugins-sk"

    pluginBT = kernel.import_semantic_skill_from_directory(pluginsDirectory, "BusinessThinking");
    my_context['input'] = 'makes pizzas'
    my_context['strengths'] = ", ".join(strengths)
    my_context['weaknesses'] = ", ".join(weaknesses)

    costefficiency_result = await kernel.run_async(pluginBT["SeekCostEfficiency"], input_context=my_context)
    print(str(costefficiency_result))

asyncio.run(main())