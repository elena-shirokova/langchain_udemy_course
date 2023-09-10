from typing import Tuple
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
from chains.custom_chains import (
    get_summary_chain,
    get_interests_chain,
    get_ice_breaker_chain,
)
from third_parties.linkedin import scrape_linkedin_profile
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import requests
from third_parties.twitter import scrape_user_tweets
from output_parsers import (
    summary_parser,
    topics_of_interest_parser,
    ice_breaker_parser,
    Summary,
    IceBreaker,
    TopicOfInterest,
)
import os
import json
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from vector_stores.faiss_db import DataLoader



def ice_break_with(name: str) -> Tuple[Summary, IceBreaker, TopicOfInterest, str]:
    linkedin_username = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username)

    summary_chain = get_summary_chain()
    summary_and_facts = summary_chain.run(
        information=linkedin_data
    )
    summary_and_facts = summary_parser.parse(summary_and_facts)

    interests_chain = get_interests_chain()
    interests = interests_chain.run(information=linkedin_data)
    interests = topics_of_interest_parser.parse(interests)

    ice_breaker_chain = get_ice_breaker_chain()
    ice_breakers = ice_breaker_chain.run(
        information=linkedin_data
    )
    
    ice_breakers = ice_breaker_parser.parse(ice_breakers)

    return (
        summary_and_facts,
        interests,
        ice_breakers,
        linkedin_data.get("profile_pic_url"),
    )


if __name__ == "__main__":
    ice_break_with("Elena Shirokova OLX")
    # linkedin_username = linkedin_lookup_agent(name="Elena Shirokova OLX")

    # summary_template = """
    #      given the information about a person from linkedin {information} I want you to create:
    #      1. a short summary
    #      2. two interesting facts about them
    #  """
    
    # # test_linkedin_data = requests.get("https ://gist.githubusercontent.com/emarco177/0d6a3f93dd06634d95e46a2782ed7490/raw/fad4d7a87e3e934ad52ba2a968bad9eb45128665/eden-marco.json")
  
    # summary_promt_template = PromptTemplate(input_variables=["information"], template=summary_template)
    # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", request_timeout=120)

    # linkedin_data = scrape_linkedin_profile(linkedin_profile_url="https://www.de.linkedin.com/in/elena-shirokova-ab3451121")
    # print(linkedin_data)

    # chain = LLMChain(llm=llm, prompt=summary_promt_template)

    # print(chain.run(information=linkedin_data))