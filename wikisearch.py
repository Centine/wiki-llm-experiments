import os
import sys
import markdown
import pandas as pd
import re
from bs4 import BeautifulSoup
from rich import print, inspect
from rich.progress import track
from time import sleep
import argparse
import nltk
from scipy import spatial
import openai
import tiktoken

# Your directory
directory = "wikijs-content"
embeddings_db_file = "data/wikijs.parquet"

GPT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"


def rcte_tokenize(data) -> list:
    sentences = []
    # Tokenize each document into sentences
    for document in track(data, description="Tokenizing documents"):
        sentences.extend(nltk.tokenize.sent_tokenize(document))

    print("Tokenization complete. Number of sentences:", len(sentences))
    return sentences


def rcte_embed_batch(sentences) -> pd.DataFrame:
    BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request

    embeddings = []
    batch_indices = range(0, len(sentences), BATCH_SIZE)
    for batch_start in track(batch_indices, description=f"Embedding sentences in batches ({BATCH_SIZE} at a time)"):
        batch = sentences[batch_start:batch_start + BATCH_SIZE]
        print(f"Batch {batch_start} to {batch_start + len(batch) - 1}")
        response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
        for i, be in enumerate(response["data"]):
            assert i == be["index"]  # double check embeddings are in same order as input
        batch_embeddings = [e["embedding"] for e in response["data"]]
        embeddings.extend(batch_embeddings)

    df = pd.DataFrame({"text": sentences, "embedding": embeddings})
    return df

# Embed each sentence, one by one. SLOOOOOW
def rcte_embed(sentences) -> pd.DataFrame:

    # Embed each sentence
    embeddings = []
    for sentence in track(sentences, description="Embedding sentences"):
        sleep(0.2)
        embedding_response = openai.Embedding.create(
            model=EMBEDDING_MODEL,
            input=sentence,
        )
        embeddings.append(embedding_response["data"][0]["embedding"])

    print("Embedding complete. Number of embeddings:", len(embeddings)) 
    # Create a DataFrame
    df = pd.DataFrame({
        'text': sentences,
        'embedding': embeddings
    })
    return df

def rcte_build_data_list(directory) -> list:
    # This will hold all your text
    datastr = ""
    datalist = []
    
    # Go through each file
    for dirpath, dirnames, filenames in track(os.walk(directory), description="Reading and parsing files"):
        for filename in filenames:
            if filename.endswith(".md"):
                # Open the file
                with open(os.path.join(dirpath, filename)) as f:
                    md_text = f.read()
                    
                    # Convert to text
                    html = markdown.markdown(md_text)
                    text = ''.join(BeautifulSoup(html, features="html.parser").findAll(text=True))
                    

                    # Add to data
                    datastr += "\n" + text
                    datalist.append(text)

    # Print some stats
    print("Prefilter, number of files:", len(datalist))
    print("Prefilter, total length of text:", len(datastr))
    return datalist

def rcte_filter(datalist) -> list:
    # Filter out metadata
    filtered_datalist = []
    filtered_datastr = ""

    for entry in track(datalist, description="Filtering out metadata"):
        # filter
        if "published: true" not in entry:
            continue
        metadata_to_remove = ["description", "date", "published", "tags", "dateCreated", "editor"]
        for metadata in metadata_to_remove:
            text = re.sub(r'{}:.*\n'.format(metadata), '', entry, flags=re.MULTILINE)
        filtered_datalist.append(text)
        filtered_datastr += "\n" + text

    # Print some stats
    print("Postfilter, number of files:", len(filtered_datalist))
    print("Postfilter, total length of text:", len(filtered_datastr))
    return filtered_datalist


def read_clean_tokenize_embed() -> pd.DataFrame:
    nltk.download('punkt')  # Download the Punkt tokenizer
    datalist = rcte_build_data_list(directory)
    filtered_datalist = rcte_filter(datalist)
    inspect(filtered_datalist)

    if filtered_datalist is None:
        print("No data found")
        sys.exit(1)

    sentences = rcte_tokenize(filtered_datalist)
    search_data = rcte_embed_batch(sentences)
    search_data.to_parquet(embeddings_db_file)
    return search_data

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# search function
def search_strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def search_test_ranked_by_relatedness(df: pd.DataFrame):
    strings, relatednesses = search_strings_ranked_by_relatedness("Authenticating to kubernetes", df, top_n=5)
    for string, relatedness in zip(strings, relatednesses):
        print(f"{relatedness=:.3f}")
        print(f"--> {string}")


def search_query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = search_strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below articles on the internal company wiki to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\Wiki article section:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def search_ask(
    query: str,
    df: pd.DataFrame,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = search_query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about the DFDS Cloud Platform."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message


def search(search_string, search_data):
    print(f"Asking GPT-3")
    print(f"Search string: {search_string}")
    print(f"Response:\n{search_ask(search_string, search_data)}")
    return



## Main
def main():
    print("Starting")
        # Parse arguments
    parser = argparse.ArgumentParser(
                        prog='wikijs-llm',
                        description='Reads, cleans, and tokenizes wikijs content, then trains a GPT-3 model on it.',
                        epilog='Text at the bottom of help')
    parser.add_argument('mode', choices=['embed', 'search', 'both'])           # positional argument
    parser.add_argument('search_str', nargs='?', default=None) # positional argument
    args = parser.parse_args()
    print("Parsed args")
    print(args)

    if args.mode == "embed":
        read_clean_tokenize_embed()
    elif args.mode == "search":
        if args.search_str is None:
            raise ValueError("Search string must be provided")
        
        search_data = pd.read_parquet(embeddings_db_file)
        search(args.search_str, search_data=search_data)
    elif args.mode == "both":
        if args.search_str is None:
            raise ValueError("Search string must be provided")
        search_data = read_clean_tokenize_embed()
        search(args.search_str, search_data=search_data)
    else:
        raise ValueError("Mode must be embed, search or both")

    print("Done")
    return

if __name__ == "__main__":
    main()
    sys.exit(0)