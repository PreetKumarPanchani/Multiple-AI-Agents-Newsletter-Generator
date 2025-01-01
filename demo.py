import os
import time
from typing import Optional
from crewai import Agent, Crew, Task, Process
from langchain_groq import ChatGroq
from tools import google_search_tool
from dotenv import load_dotenv

class AINewsletter:
    def __init__(self, topic: str, groq_api_key: Optional[str] = None):
        load_dotenv()
        self.topic = topic
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        self.llm = self._setup_llm()
        
    def _setup_llm(self):
        return ChatGroq(
            model="groq/llama-3.1-8b-instant",
            verbose=True,
            temperature=0,  # Increased for more varied responses
            groq_api_key=self.groq_api_key,
            max_tokens=300,  # Further reduced
            request_timeout=30  # Added timeout
        )

    def _create_agent(self, role, goal):
        agent = Agent(
            role=role,
            goal=goal,
            backstory=f"Expert in {role.lower()}",
            memory=False,
            verbose=True,
            llm=self.llm,
            tools=[google_search_tool],
            allow_delegation=False  # Disabled delegation
        )
        return agent

    def _create_task(self, description, expected_output, agent):
        task = Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
            max_retries=3
        )
        return task

    def generate_newsletter(self):
        try:
            # Research phase
            researcher = self._create_agent(
                "Research Analyst",
                f"Analyze trends in {self.topic}"
            )
            research_task = self._create_task(
                f"Research key trends in {self.topic}",
                "Brief trend analysis",
                researcher
            )
            
            # Writing phase
            writer = self._create_agent(
                "Content Writer",
                f"Create article about {self.topic}"
            )
            write_task = self._create_task(
                f"Write about {self.topic}",
                "Draft article",
                writer
            )
            
            # Review phase
            reviewer = self._create_agent(
                "Editor",
                f"Review content about {self.topic}"
            )
            review_task = self._create_task(
                f"Review and polish content about {self.topic}",
                "Final article",
                reviewer
            )

            crew = Crew(
                agents=[researcher, writer, reviewer],
                tasks=[research_task, write_task, review_task],
                process=Process.sequential,
                max_tokens=300,
                verbose=True
            )

            result = crew.kickoff()
            if not result:
                raise ValueError("Empty response from crew")
                
            return result

        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                print("Rate limit hit, waiting 60s...")
                time.sleep(60)
                return self.generate_newsletter()
            elif "empty response" in str(e).lower():
                print("Empty response, retrying with different parameters...")
                self.llm.temperature = 0.9  # Increase temperature
                return self.generate_newsletter()
            else:
                raise e

def main():
    topic = "Artificial Intelligence in Finance"
    retries = 3
    
    for attempt in range(retries):
        try:
            newsletter = AINewsletter(topic)
            result = newsletter.generate_newsletter()
            if result:
                print(result)
                break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < retries - 1:
                print("Retrying...")
                time.sleep(30)  # Wait between retries
            else:
                print("All attempts failed")

if __name__ == "__main__":
    main()