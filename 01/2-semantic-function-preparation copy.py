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

async def main():

    sk_prompt = """
    {{$input}}

    Summarize the content above in less than 140 characters.
    """

    summary_function = kernel.create_semantic_function(prompt_template = sk_prompt,
    description="Summarizes the input to length of an old tweet",
    max_tokens=200,
    temperature=0.1,
    top_p=0.5)

    print("A semantic function for summarization has been registered.");

    sk_input = """
    Let me illustrate an example. Many weekends, I drive a few minutes from my house to a local pizza store to buy 
    a slice of Hawaiian pizza from the gentleman that owns this pizza store. And his pizza is great, but he always 
    has a lot of cold pizzas sitting around, and every weekend some different flavor of pizza is out of stock. 
    But when I watch him operate his store, I get excited, because by selling pizza, he is generating data. 
    And this is data that he can take advantage of if he had access to AI.

    AI systems are good at spotting patterns when given access to the right data, and perhaps an AI system could spot 
    if Mediterranean pizzas sell really well on a Friday night, maybe it could suggest to him to make more of it on a 
    Friday afternoon. Now you might say to me, "Hey, Andrew, this is a small pizza store. What's the big deal?" And I 
    say, to the gentleman that owns this pizza store, something that could help him improve his revenues by a few 
    thousand dollars a year, that will be a huge deal to him.
    """;

    ## summary_result = await kernel.run_async(summary_function, input_str=sk_input)
    ## above is one indirect way to invoke the function

    ## this is short way to call the summary function
    summary_result = summary_function(sk_input)

    print(str(summary_result))

asyncio.run(main())