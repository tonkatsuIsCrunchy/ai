import requests
import os
import re

import html2text
from bs4 import BeautifulSoup
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_text_splitters import CharacterTextSplitter

from pydantic import BaseModel, Field, create_model


def fetch_html(url):
    raw_html = requests.get(url, verify=False)
    return raw_html.text


def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove headers and footers based on common HTML tags or classes
    for element in soup.find_all(['header', 'footer']):
        element.decompose()  # Remove these tags and their content

    return str(soup)


def html_to_markdown_with_readability(html_content):
    cleaned_html = clean_html(html_content)  
    
    # Convert to markdown
    markdown_converter = html2text.HTML2Text()
    markdown_converter.ignore_links = False
    markdown_content = markdown_converter.handle(cleaned_html)
    
    return markdown_content

def save_raw_data(raw_data, timestamp, output_folder='output'):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the raw markdown data with timestamp in filename
    raw_output_path = os.path.join(output_folder, f'rawData_{timestamp}.md')
    with open(raw_output_path, 'w', encoding='utf-8') as f:
        f.write(raw_data)
    print(f"Raw data saved to {raw_output_path}")
    return raw_output_path


###############
def extract_urls_from_markdown(markdown_text):
    """
    Extract URLs from a given markdown text using a regex pattern.
    """
    url_pattern = re.compile(r'https?://\S+')
    return url_pattern.findall(markdown_text)

def split_dom_content(dom_content, max_length=1000):
    text_splitter = CharacterTextSplitter(separator="\n\n",chunk_size=max_length, chunk_overlap=20)
    chunks = text_splitter.split_text(dom_content)

    return chunks


def split_dom_content2(dom_content, max_length=6000): #split content for token limit of llm
    return [
        dom_content[i : i + max_length] for i in range(0, len(dom_content), max_length)
    ]


def format_data(fields, dom_chunks):
    model = OllamaLLM(model="llama3.2")
    #model = OllamaLLM(model="mistral")
    #model = OllamaLLM(model="codellama")

    template = (
        "You are tasked with extracting specific information from the following text content: {dom_content}. "
        "Please follow these instructions carefully: \n\n"
        "1. **Extract Information:** Only extract the information that directly matches the provided description: {fields}"
        "2. **No Extra Content:** Do not include any additional text, comments, or explanations in your response. "
        "3. **Empty Response:** If no information matches the description, return an empty string ('')."
        "4. **Direct Data Only:** Your output should contain only the data that is explicitly requested, with no other text."
    )

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    print("invoking llm...\n")

    parsed_results = []
    for i, chunk in enumerate(dom_chunks, start=1):
        response = chain.invoke({"dom_content": chunk, "fields": fields})
        print(f"Parsed batched {i} of {len(dom_chunks)}")
        parsed_results.append(response)
        print("---")
        for item in parsed_results:
            print("++", item)
        print("---")
        
    #return "\n".join(parsed_results)
    return parsed_results

#    response = chain.invoke({"dom_content": data})

#    return response


###############


def main():
    url = r"https://sg.finance.yahoo.com/topic/stocks/"
    fields = ["urls starting with http:// or https://"]
    
    print("enter url:")
    url = input()
    print("enter fields to extract:")
    fields = input()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    raw_html = fetch_html(url)
    print(raw_html)

    markdown = html_to_markdown_with_readability(raw_html)

    # Save raw data
    save_raw_data(markdown, timestamp)

    """
    print("Extract from REGEX")
    urls = extract_urls_from_markdown(markdown)
    print(f"Extracted urls (regex), total={len(urls)}:")
    i = 0
    for item in urls:
        i = i + 1
        print(i, item)


    print("\n\n")
    """

    print("Extract from LLM")
    dom_chunks = split_dom_content(markdown)
    result = format_data(fields, dom_chunks)
    i = 0
    for item in result:
        i = i + 1
        print(f"{i}=", item)


if __name__ == "__main__":
    main()

