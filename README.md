# Developing-a-chat-model-

# ML/AI Shortlisting Task - 1

 Installing two Python packages Langchain and Openai

!pip install -qU langchain openai

We'll start by initializing the ChatOpenAI object. For this we'll need an OpenAI API key.

from getpass import getpass

# enter your api key
OPENAI_API_KEY = getpass("OpenAI API key: ")

from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    model='gpt-3.5-turbo'
)

 Importing the ChatOpenAI class from the langchain.chat_models module and creating an instance of it. 
 The argument temperature determines the randomness of the model's responses. A temperature of 0 indicates that the model will always generate the most likely response. A higher temperature value (e.g., 0.8) produces more diverse and creative responses, while a lower value (e.g., 0.2) produces more focused and deterministic responses. 

The argument model specifies the specific version or variant of the language model to use. In this case, the model is set to 'gpt-3.5-turbo', which refers to OpenAI's GPT-3.5 Turbo model. This model is known for its ability to generate high-quality text and is optimized for a balance of speed and quality.

Creating a conversation or dialogue by defining a sequence of messages

from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi AI, how are you today?"),
    AIMessage(content="I'm great thank you. How can I help you?"),
    HumanMessage(content="I'd like to understand string theory.")
]

The messages include a system message, a message from the human, and a message from the AI assistant.

res = chat(messages)
res

In response we get another AI message object. We can print it more clearly like so:

print(res.content)

Because res is just another AIMessage object, we can append it to messages, add another HumanMessage, and generate the next response in the conversation.

# add latest AI response to messages
messages.append(res)

# now create a new user prompt
prompt = HumanMessage(
    content="Why do physicists believe it can produce a 'unified theory'?"
)
# add to messages
messages.append(prompt)

# send to chat-gpt
res = chat(messages)

print(res.content)

chat = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    model='gpt-3.5-turbo-0301'
)

# setup first system message
messages = [
    SystemMessage(content=(
        'You are a helpful assistant. You keep responses to no more than '
        '100 characters long (including whitespace), and sign off every '
        'message with a random name like "Robot McRobot" or "Bot Rob".'
    )),
    HumanMessage(content="Hi AI, how are you? What is quantum physics?")
]

The messages represents the conversation between the user and the AI assistant. Sets up the initial conversation with the AI assistant, providing the necessary context for the AI to generate a relevant response. 

Continues the conversation with the AI assistant. The response from the assistant is stored in the res variable.

res = chat(messages)

print(f"Length: {len(res.content)}\n{res.content}")

Okay, seems like our AI assistant isn't so good at following either of our instructions. What if we add these instructions to the HumanMessage via a HumanMessagePromptTemplate?

Alongside what we've seen so far there are also three new prompt templates that we can use. Those are the SystemMessagePromptTemplate, AIMessagePromptTemplate, and HumanMessagePromptTemplate. The prompt template classes in Langchain are built to make constructing prompts with dynamic inputs easier.

These are simply an extension of Langchain's prompt templates that modify the returning "prompt" to be a SystemMessage, AIMessage, or HumanMessage object respectively.

from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate

human_template = HumanMessagePromptTemplate.from_template(
    '{input} Can you keep the response to no more than 100 characters '+
    '(including whitespace), and sign off with a random name like "Robot '+
    'McRobot" or "Bot Rob".'
)

# create the human message
chat_prompt = ChatPromptTemplate.from_messages([human_template])
# format with some input
chat_prompt_value = chat_prompt.format_prompt(
    input="Hi AI, how are you? What is quantum physics?"
)
chat_prompt_value

The ChatPromptTemplate is used to generate a prompt for a chat-based model. By providing the human_template, it indicates that the conversation will start with a human message. The format_prompt method performs the substitution and returns the formatted prompt. Here, templates to create a human message prompt template, formats it with a specific input, and stores the formatted prompt value. This allows for the customization of prompts used in chat-based models, providing flexibility in generating conversations with AI assistants.

Using this we return a ChatPromptValue object. This can be formatted into a list or string like so:

chat_prompt_value.to_messages()

chat_prompt_value.to_string()

Let's see if this new approach works.

messages = [
    SystemMessage(content=(
        'You are a helpful assistant. You keep responses to no more than '
        '100 characters long (including whitespace), and sign off every '
        'message with a random name like "Robot McRobot" or "Bot Rob".'
    )),
    chat_prompt.format_prompt(
        input="Hi AI, how are you? What is quantum physics?"
    ).to_messages()[0]
]

res = chat(messages)

print(f"Length: {len(res.content)}\n{res.content}")

This time we get pretty close, we're slightly over the character limit (by 8 characters), and we got a sign off with - Bot Rob. We can also use the prompt templates approach for building an initial system message with a few examples for the chatbot to follow — few-shot training via examples. Let's see what that looks like.

Sets up a conversation by defining a list of messages containing a system message and an initial user prompt. It then passes this list to the chat function to obtain a response from the AI assistant. Finally, it prints the length and content of the response.

from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate
)

system_template = SystemMessagePromptTemplate.from_template(
    'You are a helpful assistant. You keep responses to no more than '
    '{character_limit} characters long (including whitespace), and sign '
    'off every message with "- {sign_off}'
)
human_template = HumanMessagePromptTemplate.from_template("{input}")
ai_template = AIMessagePromptTemplate.from_template("{response} - {sign_off}")

# create the list of messages
chat_prompt = ChatPromptTemplate.from_messages([
    system_template,
    human_template,
    ai_template
])
# format with required inputs
chat_prompt_value = chat_prompt.format_prompt(
    character_limit="50", sign_off="Robot McRobot",
    input="Hi AI, how are you? What is quantum physics?",
    response="Good! It's physics of small things"
)
chat_prompt_value

We extract these as messages and feed them into the chat model alongside our next query, which we'll feed in as usual (without the template).

messages = chat_prompt_value.to_messages()

messages.append(
    HumanMessage(content="How small?")
)

res = chat(messages)

print(f"Length: {len(res.content)}\n{res.content}")

Perfect, we seem to get a good response there, let's try a couple more

# add last response
messages.append(res)

# make new query
messages.append(
    HumanMessage(content="Okay cool, so it is like 'partical physics'?")
)

res = chat(messages)

print(f"Length: {len(res.content)}\n{res.content}")

We went a little over here. We could begin using the previous HumanMessagePromptTemplate again like so:

from langchain import PromptTemplate

# this is a faster way of building the prompt via a PromptTemplate
human_template = HumanMessagePromptTemplate.from_template(
    '{input} Answer in less than {character_limit} characters (including whitespace).'
)
# create the human message
human_prompt = ChatPromptTemplate.from_messages([human_template])
# format with some input
human_prompt_value = human_prompt.format_prompt(
    input="Okay cool, so it is like 'partical physics'?",
    character_limit="50"
)
human_prompt_value


# drop the last message about partical physics so we can rewrite
messages.pop(-1)

messages.extend(human_prompt_value.to_messages())
messages

Now process:

res = chat(messages)

print(f"Length: {len(res.content)}\n{res.content}")

There we go, a good answer again!

Now, it's arguable as to whether all of the above is better than simple f-strings like:

_input = "Okay cool, so it is like 'partical physics'?"
character_limit = 50

human_message = HumanMessage(content=(
    f"{_input} Answer in less than {character_limit} characters "
    "(including whitespace)."
))

human_message

In this example, the above is far simpler. So we wouldn't necessarily recommend using prompt templates over f-strings in all scenarios. But, if you do find yourself in a scenario where they become more useful — you now know how to use them.

To finish off, let's see how the rest of the completion process can be done using the f-string formatted message human_message:

# drop the last message about partical physics so we can rewrite
messages.pop(-1)

# add f-string formatted message
messages.append(human_message)
messages

res = chat(messages)

print(f"Length: {len(res.content)}\n{res.content}")

That's it for this example exploring LangChain's new features for chat.
