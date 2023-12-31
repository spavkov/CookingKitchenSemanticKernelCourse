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

    swot_interview= """
    1. **Strengths**
        - What unique recipes or ingredients does the pizza shop use?
        - What are the skills and experience of the staff?
        - Does the pizza shop have a strong reputation in the local area?
        - Are there any unique features of the shop or its location that attract customers?
    2. **Weaknesses**
        - What are the operational challenges of the pizza shop? (e.g., slow service, high staff turnover)
        - Are there financial constraints that limit growth or improvements?
        - Are there any gaps in the product offering?
        - Are there customer complaints or negative reviews that need to be addressed?
    3. **Opportunities**
        - Is there potential for new products or services (e.g., catering, delivery)?
        - Are there under-served customer segments or market areas?
        - Can new technologies or systems enhance the business operations?
        - Are there partnerships or local events that can be leveraged for marketing?
    4. **Threats**
        - Who are the major competitors and what are they offering?
        - Are there potential negative impacts due to changes in the local area (e.g., construction, closure of nearby businesses)?
        - Are there economic or industry trends that could impact the business negatively (e.g., increased ingredient costs)?
        - Is there any risk due to changes in regulations or legislation (e.g., health and safety, employment)?"""

    sk_prompt_for_shift_domain = """
    {{$input}}

    Convert the analysis provided above to the business domain of {{$domain}}.
    """
    shift_domain_function = kernel.create_semantic_function(prompt_template = sk_prompt_for_shift_domain,
    description="Translates an idea into another domain",
    max_tokens=1000,
    temperature=0.1,
    top_p=0.5)

    print("A semantic function shift_domain_function has been registered.");


    sk_prompt = """
    {{$input}}

    Rewrite the text above to be understood even by a {{$level}}.
    """

    shift_reading_level_function = kernel.create_semantic_function(prompt_template=sk_prompt,
                                                        description="Change the reading level of a given text.",
                                                        max_tokens=1000,
                                                        temperature=0.1,
                                                        top_p=0.5)

    print("A semantic function shift_reading_level_function has been registered.");

    my_context['input'] = swot_interview
    my_context['domain'] = "construction management"
    my_context['level'] = "preschool child that does not know anything about business"

    result = await kernel.run_async(shift_domain_function, shift_reading_level_function, input_context=my_context)

    print(str(result))

asyncio.run(main())