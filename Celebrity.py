##here we integrate our code with OPEN AI api
import os
from constants import openai_key
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

import streamlit as sl


#streamlit framework

sl.title('Langchain demo')
text=sl.text_input("Search")

#prompt template

fip=PromptTemplate(
    input_variables=['celebrity'],
    template="Tell me about {celebrity}"
)

#we initialize the llm now
llm = ChatOpenAI(openai_api_key=openai_key)
chain=LLMChain(llm=llm,prompt=fip,verbose=True,output_key='person')


#second prompt template
sip=PromptTemplate(
    input_variables=['person'],
    template="when was {person} born"
)



chain2=LLMChain(llm=llm,prompt=sip,verbose=True,output_key='dob')

#Third Template
tip=PromptTemplate(
    input_variables=['dob'],
    template="What is special about {dob}"
)
chain3=LLMChain(llm=llm,prompt=tip,verbose=True,output_key='desc')


#combining all the chains
parent=SequentialChain(chains=[chain,chain2,chain3],input_variables=['celebrity'],output_variables=['person','dob','desc'],verbose=True)


if text:
    sl.write(parent({'celebrity':text}))
