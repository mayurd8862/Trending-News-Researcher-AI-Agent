# import os
# from crewai import Agent, Task, Crew, LLM
# from crewai_tools import SerperDevTool, WebsiteSearchTool, FileReadTool, YoutubeVideoSearchTool
# from dotenv import load_dotenv
# import time
# load_dotenv()

# # Set up API keys
# os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")  # serper.dev API key
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# llm = LLM(
#     model="groq/gemma2-9b-it",
#     temperature=0.5
# )

# # Instantiate tools
# search_tool = SerperDevTool()
# file_tool = FileReadTool()

# # Create agents
# news_researcher = Agent(
#     role='Trending News Researcher',
#     goal='Fetch and summarize the latest trending news for {topic}.',
#     backstory='An AI-driven journalist, passionate about uncovering the latest news trends.',
#     tools=[search_tool],
#     verbose=True,
#     max_tokens=5000,
#     max_iter= 10,
#     max_rpm=28,
#     llm = llm
# )

# news_writer = Agent(
#     role='Content Writer',
#     goal='Craft an engaging blog post based on the latest trending news for {topic}.',
#     backstory='A skilled writer with a passion for current events.',
#     tools=[file_tool],
#     verbose=True,
#     max_tokens=5000,
#     max_iter= 10,
#     max_rpm=28,
#     llm = llm
# )

# # Define tasks
# research_task = Task(
#     description='Fetch the latest trending news for the given topic {topic} and provide a summary.',
#     expected_output='A summary of the top 3 trending news articles for the given topic {topic} with a unique perspective on their significance.',
#     agent=news_researcher
# )

# write_task = Task(
#     description='Write an engaging blog post about the trending news, based on the researcher’s summary.',
#     expected_output='A 4-paragraph blog post formatted in markdown with engaging, informative, and accessible content, avoiding complex jargon.',
#     agent=news_writer,
#     output_file='blog-posts/new_post.md'  # The final blog post will be saved here
# )

# # Function to execute the crew process with user input
# def main(topic):
#     # Assemble a crew with planning enabled
#     crew = Crew(
#         agents=[news_researcher, news_writer],
#         tasks=[research_task, write_task],
#         verbose=True,
#     )
    
#     # Kick off the process with the topic as input
#     result = crew.kickoff(inputs={'topic': topic})
#     print(result)

# if __name__ == "__main__":
#     user_topic = input("Enter the topic for trending news: ")
#     main(user_topic)


import os
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check API key
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
if not SERPER_API_KEY:
    raise ValueError("SERPER_API_KEY is missing from .env file!")

# Configure LLM
llm = LLM(
    model="groq/llama-3.1-8b-instant",  # Or any other suitable model
    temperature=0.5
)

# Explicitly create and configure the search tool
search_tool = SerperDevTool(api_key=SERPER_API_KEY)

# Improved Agent and Task definitions: crucial for tool usage
news_agent = Agent(
    role='News Researcher',
    goal='Fetch and summarize the latest trending news for {topic}.',
    backstory='An AI journalist skilled at web research.',
    tools=[search_tool],
    verbose=True,
    llm=llm
)

research_task = Task(
    description=f"""
    You are a News Researcher. Your task is to find the latest news about the given topic: {{topic}}.

    To accomplish this, you MUST use the available tool: `serper_search`.

    Here's how to use the tool:
    ```
    serper_search(query="{{topic}}")
    ```
    This will give you search results. Use these results to create a detailed summary of the top 3 trending news articles.

    Your final response should be a summary of the top 3 articles, including their titles and brief descriptions.
    """,
    expected_output='A detailed summary of the top 3 trending news articles.',
    agent=news_agent
)


# Main function to execute the crew
def main(topic):
    crew = Crew(
        agents=[news_agent],
        tasks=[research_task],
        verbose=True
    )

    try:
        result = crew.kickoff(inputs={'topic': topic})
        print("Search Result:", result)
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    topic = input("Enter a topic to search: ")
    main(topic)