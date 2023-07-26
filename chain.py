import os
import pinecone

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv())


def make_chain(pinecone_index: str) -> ConversationalRetrievalChain:
    # initialize model
    model = ChatOpenAI(temperature=0.0)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                       model_kwargs={'device': 'cpu'},
                                       encode_kwargs={'normalize_embeddings': True})
    pinecone.init(api_key=os.environ['PINECONE_API_KEY'],
                  environment=os.environ['PINECONE_ENV'])

    vector_store = Pinecone.from_existing_index(pinecone_index, embeddings)

    template = """You are a helpful AI assistant that answers questions about an e-commerce company called "Sindabad.com" / 
    in a friendly and polite manner. You will be given a "Context" that will represent Sindabad.com's product inventory./ 
    Users might ask about products, they might want to know your suggestions as well. Most importantly, they might ask about specific product and/ 
    its associated product link. If they want to know about product links, you will provide it accordingly with the /
    help of the given "Context". It should be noted, only the Mobile Phones sub category has product links in them,/ 
    therefore, should the users ask a different product and want to know their product links you should tell them,/ 
    "Only the mobile phones section has product links at the moment, in future additional sections will have product/ 
    links in them". Answer the question in your own words as truthfully as possible from the context given to you./ 
    If you do not know the answer to the question, simply respond with "I don't know. Could you please rephrase the/ 
    question?". If questions are asked where there is no relevant information available in the context / 
    , answer the question with your existing knowledge on that question and "ignore" the "Context" given to you. / 
    Lastly, if the user thanks you for your response, respond with, "You're welcome! Thank you for choosing Sindabad.com"
    
    Context:\n {context}
    
    Human: {question}
    Assistant:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    condense_prompt = PromptTemplate(
        input_variables=["question", "chat_history"],
        template="""Combine the chat history and follow up question into /
        a standalone question. /
        Chat History: {chat_history}
        Follow up question: {question}
        """
    )
    memory = ConversationBufferWindowMemory(input_key="question",
                                            memory_key="chat_history",
                                            k=3,
                                            return_messages=True,
                                            output_key="answer")
    return ConversationalRetrievalChain.from_llm(model,
                                                 retriever=vector_store.as_retriever(),
                                                 return_source_documents=True,
                                                 condense_question_prompt=condense_prompt,
                                                 combine_docs_chain_kwargs={
                                                     'prompt': prompt},
                                                 memory=memory,
                                                 verbose=False)
