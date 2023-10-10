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

class ExoticLanguagePlugin:
    def word_to_pig_latin(self, word):
        vowels = "AEIOUaeiou"
        if word[0] in vowels:
            return word + "way"
        for idx, letter in enumerate(word):
            if letter in vowels:
                break
        else:
            return word + "ay"
        return word[idx:] + word[:idx] + "ay"
    @sk_function(
        description="Takes text and converts it to pig latin",
        name="pig_latin",
        input_description="The text to convert to pig latin",
    )
    def pig_latin(self, sentence:str) -> str:
        words = sentence.split()
        pig_latin_words = []
        for word in words:
            pig_latin_words.append(self.word_to_pig_latin(word))
        return ' '.join(pig_latin_words)


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
    You can use the following tips to help you become a better programmer:

    Write each line of code multiple times. Repetition can reduce coding errors and help you learn to recognize similarities among codes. Some programmers also use the rule of three when coding, which means they write each code sequence three times to decide when to delete code that's too similar.
    Practice code in different environments. If you're familiar with writing code for a certain industry, try coding in a different industry. You might also practice code beyond school or work assignments and work on projects you're less familiar with to help improve your skills.
    Practice unit testing. Consider practicing unit testing, which is the process of checking a small piece of code and identifying how it affects other areas of code. This method can also help with identifying coding errors before finishing a project.
    Improve soft programming skills. Improving your soft skills can help you develop your programming style and work with other programmers. A few important soft programming skills include teamwork, communication and project management.
    Sign up for newsletters. Staying informed on the latest trends and updates in the industry is an important part of becoming a better programmer. Newsletters are one way you can learn about relevant industry changes.
    Read code. Becoming a better programmer involves both practicing code and reading code to learn how to better identify broken code. Inspect the source code of your favorite websites or review previous code you've written.
    Rewrite your code when you learn something new. As soon as you complete a programming class or expand your coding skills to include a new language, consider rewriting previous code. This may help you identify techniques you can improve.
    Contribute to the open-source community. Contributing to an open-source community, or collection of programming contributors, can allow you to assist other programmers and develop your programming skills. It also allows other programmers to view your code and offer you specific feedback.
    """;

    exotic_language_plugin = kernel.import_skill(ExoticLanguagePlugin(), skill_name="exotic_language_plugin")
    pig_latin_function = exotic_language_plugin["pig_latin"]

    print("registered pig latin function")

    # this code now creates pipeline which first calls summary_function with our input
    # and then pipes this into pig lating function and then returns result
    # that way we can combine the native c# or python functions with the semantic functions that use LLM
    final_result = await kernel.run_async(summary_function, pig_latin_function, input_str=sk_input) 

    print(str(final_result))

asyncio.run(main())