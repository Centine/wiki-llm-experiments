import os
import sys
import markdown
import pandas as pd
import re
from bs4 import BeautifulSoup
from rich import print, inspect
from rich.pretty import pprint
from rich.progress import track
from time import sleep
import argparse
import nltk
from scipy import spatial
import openai
import tiktoken

def filter_doc(entry) -> str:
    # Filter out metadata
    filtered_entry = ""
    # filter
    metadata_to_remove = ["description", "date", "published", "tags", "dateCreated", "editor"]
    for metadata in metadata_to_remove:
        text = re.sub(r'{}:.*\n'.format(metadata), '', entry, flags=re.MULTILINE)
    filtered_entry = text
    return filtered_entry


def generate_summary(sentences: list):
    output_chunks = []
    for chunk in track(sentences):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=(f"Please summarize the following text:\n{chunk}\n\nSummary:"),
            temperature=0.5,
            max_tokens=1024,
            n = 1,
            stop=None
        )
        summary = response.choices[0].text.strip()
        output_chunks.append(summary)
    return " ".join(output_chunks)

def rcte_tokenize(document: str) -> list:
    sentences = []
    # Tokenize each document into sentences
    
    sentences.extend(nltk.tokenize.sent_tokenize(document))

    print("Tokenization complete. Number of sentences:", len(sentences))
    return sentences

def summarize(filename, mode: str = 'monoblock'):        
    with open(secure_filename(filename=filename)) as f:
        md_text = f.read()
    # deepcode ignore PT: has been validated
    
        # Convert to text
        html = markdown.markdown(md_text)
        text = ''.join(BeautifulSoup(html, features="html.parser").findAll(text=True))

    print("Text length:", len(text))
    inspect(text)    

    filtered_text = filter_doc(text)
    summary = ""
    match mode:
        case 'monoblock':
            print("Monoblock mode")
            summary = generate_summary([filtered_text])            
        case 'tokenized':
            print("Tokenized mode")
            sentences = rcte_tokenize(filtered_text)
            summary = generate_summary(sentences=sentences)
        case 'summarize': 
            print("Summarize (openai-summarize) mode")
            import openai_summarize
            openai_summarizer = openai_summarize.OpenAISummarize(os.environ['OPENAI_API_KEY'])
            summary = openai_summarizer.summarize_text(text=filtered_text)

        case _: 
            raise ValueError(f"Unknown mode {mode}")        

    print(f"Summary mode '{mode}'. Summary:\n{summary}")    

def secure_filename(filename: str) -> str:
    # Check if file is in the current directory or its subdirectories
    if not os.path.abspath(filename).startswith(os.getcwd()):
        raise ValueError(f"{filename} is not in the current directory or its subdirectories.")
        
    # Check if file has a .md extension
    if not filename.endswith('.md'):
        raise ValueError(f"{filename} is not a Markdown file.")
    
    # Check if file is a symlink
    if os.path.islink(filename):
        raise ValueError(f"{filename} is a symlink.")
    
    # Check if file is a directory
    if os.path.isdir(filename):
        raise ValueError(f"{filename} is a directory.") 
    
    # check if file exists
    if not os.path.exists(filename):
        raise ValueError(f"{filename} does not exist.")
    
    # Check if filename containts illegal characters, such as two dots in a row
    if '..' in filename:
        raise ValueError(f"{filename} contains illegal characters.")
    
    return str(filename)

## Main
def main():
    print("Starting")
        # Parse arguments
    parser = argparse.ArgumentParser(
                        prog='wikisummarize',
                        description='Summarize wikijs articles',
                        epilog='Text at the bottom of help')
    parser.add_argument('filename')
    parser.add_argument('--mode', default='monoblock', help='Mode: monoblock, summarize or tokenized', choices=['monoblock', 'summarize', 'tokenized'])
    args = parser.parse_args()
    print("Parsed args")
    print(args)

    summarize(args.filename, args.mode)

    print("Done")
    return

if __name__ == "__main__":
    main()
    sys.exit(0)