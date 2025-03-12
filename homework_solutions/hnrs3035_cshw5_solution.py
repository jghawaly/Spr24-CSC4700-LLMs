import os
import json

from openai import OpenAI
from dotenv import load_dotenv
import argparse
import chromadb
from chromadb.utils import embedding_functions
from pyowm import OWM

from prompts import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run a simple GPT-4o-mini that can answer questions about squad v2 dataset, summarize things, and check the weather")
    parser.add_argument('q', type=str, nargs="+", help="the query/question")
    parser.add_argument('--f', type=str, default=None, help="path to file query (.txt only)")
    parser.add_argument('--db_path', type=str, default='../data/squad_db',
                        help='path to where the database will be stored')
    parser.add_argument('--k', type=int, default=5, help='number of context chunks to retrieve for RAG')
    args = parser.parse_args()

    user_query = args.q
    file_query_content = None
    if args.f is not None:
        with open(args.f, 'r') as f:
            file_query_content = f.read()
            user_query += file_query_content

    # load environmental variables
    load_dotenv('../.env')

    # Create our OpenAI client
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # establish ChromaDB client
    chroma_client = chromadb.PersistentClient(path=args.db_path)

    # setup embeddings
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )

    # get the documents collection, with a provided embedding function
    collection = chroma_client.get_collection(
        name="documents",
        embedding_function=openai_ef
    )

    # created our messages list.
    messages = [{"role": "system", "content": tool_selector_prompt_1.format(question=user_query)}, ]

    response = openai_client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        temperature=0.0
    )
    tool_response = json.loads(response.choices[0].message.content)

    tool_name = tool_response["tool_name"]
    tool_args = tool_response["args"]
    print(f"Selected tool: {tool_name}")
    for arg in tool_args:
        print(f"\t{arg}: {tool_args[arg]}")
    response = ""
    if "rag" in tool_name:
        # retrieve context using embeddings
        retrieved_context = collection.query(query_texts=[tool_args['query']], n_results=args.k)
        formatted_context = "Chunk\n" + "\n\nChunk\n".join([doc[0] for doc in retrieved_context['documents']])

        # created our messages list.
        messages = [{"role": "system", "content": rag_prompt_1.format(question=tool_args['query'],
                                                                      context=formatted_context)}, ]

        response = openai_client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            temperature=0.0
        )

        response = response.choices[0].message.content
    elif "summarize" in tool_name:
        # created our messages list.
        messages = [{"role": "system", "content": chain_of_density_prompt.format(content=file_query_content if file_query_content is not None else user_query)}, ]

        response = openai_client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            temperature=0.0
        )

        response = json.loads(response.choices[0].message.content)

        print(f"Full Response: {response}")

        response = response['summaries'][-1]['Denser_Summary']
    elif "weather" in tool_name:
        my_owm = OWM(os.getenv('OMW_API_KEY'))
        mgr = my_owm.weather_manager()
        reg = my_owm.city_id_registry()
        mgr = my_owm.weather_manager()

        # get the ID for the location
        id = reg.locations_for(tool_args['city'], state=tool_args['state'], country=tool_args['country'], matching='exact')[0].id

        # get the weather for that ID
        weather = mgr.weather_at_id(id).weather

        # get some stuff we want
        weather_str = f'Temperature = {weather.temperature("fahrenheit")["temp"]} F, Feels Like Temperature = {weather.temperature("fahrenheit")["feels_like"]} F, Humidity = {weather.humidity} %, Detailed Status = {weather.detailed_status}, Wind = {weather.wind(unit="miles_hour")["speed"]} mph at {weather.wind(unit="miles_hour")["deg"]} degrees'

        # created our messages list.
        messages = [{"role": "system", "content": omw_output_format_prompt.format(omw_tool_output=weather_str,
                                                                                  query=user_query)},
                    ]

        response = openai_client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            temperature=0.0
        )

        response = response.choices[0].message.content

    print(f'\n\nResponse:\n{response}')
