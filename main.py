from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
import os
from dotenv import load_dotenv
import time
load_dotenv()

llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.7
)


os.environ["SERPER_API_KEY"] = os.environ.get("SERPER_API_KEY") # serper.dev API key
search_tool = SerperDevTool()

9168686151
# Agent definition

researcher = Agent(
    role="{topic} Senior Researcher",
    goal="""Uncover groundbreaking technologies in
    {topic} for year 2024""",
    backstory="""Driven by curiosity, you explore and
    share the latest innovations.""",
    # max_iter=15,
    # max_rpm=28,
    # verbose=True,
    tools=[search_tool],
    llm=llm
)


research_task = Task(
    description="""Identify the next big trend in
    {topic} with pros and cons.""",
    expected_output="""A 3-paragraph report on emerging
    {topic} technologies.""",
    agent=researcher
)

def main():
    # Forming the crew and kicking off the process
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True
    )
    result = crew.kickoff(inputs={'topic': 'AI'})
    print(result)


if __name__ == "__main__":
    main()