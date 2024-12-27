from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, FileWriterTool
from datetime import datetime
import os

@CrewBase
class AiNews():
    """AiNews crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self):
        # Initialize Cohere LLM configuration
        self.llm = LLM(
			provider="cohere",
			model="command-r-plus-08-2024",
			temperature=0.7,
			api_key=os.getenv('COHERE_API_KEY'),
			context_window=2048,  # Explicitly set context window
			max_tokens=500,       # Explicitly set max tokens
		)


    @agent
    def retrieve_news(self) -> Agent:
        return Agent(
            config=self.agents_config['retrieve_news'],
            tools=[SerperDevTool()],
            verbose=True,
            llm=self.llm  # Explicitly set LLM
        )
    
    @agent
    def ai_news_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['ai_news_writer'],
            tools=[],
            verbose=True,
            llm=self.llm  # Explicitly set LLM
        )
    
    @agent
    def file_writer(self) -> Agent:
        output_file = f"{datetime.now().strftime('%Y-%m-%d')}_news_article.txt"
        return Agent(
            config=self.agents_config['file_writer'],
            tools=[FileWriterTool()],
            verbose=True,
            llm=self.llm,  # Explicitly set LLM
        )

    @agent
    def website_scraper(self) -> Agent:
        return Agent(
            config=self.agents_config['website_scraper'],
            tools=[FileWriterTool()],
            verbose=True,
            llm=self.llm  # Explicitly set LLM
        )

    @task
    def retrieve_news_task(self) -> Task:
        return Task(
            config=self.tasks_config['retrieve_news_task'],
        )

    @task
    def website_scrape_task(self) -> Task:
        return Task(
            config=self.tasks_config['website_scrape_task'],
        )

    @task
    def ai_news_write_task(self) -> Task:
        return Task(
            config=self.tasks_config['ai_news_write_task'],
        )

    @task
    def file_write_task(self) -> Task:
        return Task(
            config=self.tasks_config['file_write_task'],
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the AiNews crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
