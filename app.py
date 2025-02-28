from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

def main():
    #llm = OllamaLLM(model="deepseek-r1:8b")
    llm = OllamaLLM(model="deepseek-r1:1.5b")

    response = llm.invoke("What is the capital of France?")
    print(response)

def test2():
    template = """
        Question: {question}
        Skip and hide the thinking process.
        Provide the answer immediately.

        Answer: 
        """

    prompt = ChatPromptTemplate.from_template(template)

    model = OllamaLLM(model="llama3.2")
    chain = prompt | model
    response = chain.invoke({"question": "What is the capital of France?"})
    print(response)


def test3():
    template = """
        Question: {question}

        Answer: 
        """

    prompt = ChatPromptTemplate.from_template(template)

    sysmessage = """
        Table structures:
        - Person (person_id, name, address)
        - Company (id, company_name, address)
        - Works_in (id, company_id, job_title)

        Return a SQL statement only for response.

    """
    messages = [
            SystemMessage(f"Respond with the following table structure as reference {sysmessage}."),
            HumanMessage("How many persons work as engineers?"),
    ]

    model = OllamaLLM(model="llama3.2")
    #model = OllamaLLM(model="deepseek-r1:1.5b")

    chain = prompt | model
    response = chain.invoke(messages)
    #response = model.invoke(messages)
    print(response)

if __name__ == "__main__":
    #main()
    #test2()
    test3()

