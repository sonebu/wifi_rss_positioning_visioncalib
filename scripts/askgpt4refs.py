### influenced from: https://platform.openai.com/docs/assistants/tools/file-search/quickstart
### python askgpt4refs.py -p "Briefly summarize the main contributions of all papers separately. Make sure you give a summary for each paper (do not skip any of the papers in your knowledge base)." -v vs_NIDyAs0elCdzKXfzNa7d8bCF

import argparse, os, glob
from openai import OpenAI

def parse_arguments():
    parser = argparse.ArgumentParser(description='Make vector stores out of reference PDF files under doc/references/, send them to OpenAI, ask questions about the papers to GPT-4o')
    parser.add_argument('-i','--instructions', help='Enter a string here if you want to use a custom model instruction', required=False)
    parser.add_argument('-p','--prompt',       help='Enter your prompt string here', required=True)
    parser.add_argument('-f','--folderpath',   help='If you want to ask questions to only papers grouped under a specific folder, then enter the relative path string of the folder (starting from the path of this script). The path must end with a / character to denote that it is a folder. Default folder is /doc/references/', required=False)
    parser.add_argument('-v','--vectorstoreid',help='', required=False)
    args = parser.parse_args()
    return args

def main(args):
    print("[api-state]: started script")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if(args.instructions is not None):
        instructions = args.instructions
    else:
        instructions= """You are an expert academic researcher working on WiFi sensing and AI. Use your knowledge base to answer questions about the content of academic papers."""

    assistant = client.beta.assistants.create(name="WiFi Sensing References Scanning Assistant", instructions=instructions, model="gpt-4o", tools=[{"type": "file_search"}])
    print("[api-state]: created assistant")

    if(args.vectorstoreid is not None):
        vector_store_id = args.vectorstoreid
        print("[api-state]: re-using vector store with ID "+vector_store_id)
    else:
        vector_store = client.beta.vector_stores.create(name="WiFi Sensing References")
        print("[api-state]: created vector store with ID "+vector_store.id)
        print("             please use this ID with the argument -v for your next question if you want to ask a different question to the same set of PDFs.")
        print("")

        if(args.folderpath is not None):
            file_paths = glob.glob(args.folderpath+"*.pdf", recursive=True)
        else:
            file_paths = glob.glob("../docs/references/**/*.pdf", recursive=True)
        file_streams = [open(path, "rb") for path in file_paths]

        print("[api-state]: the following files are going to be added to the vector store:")
        print("")
        for path in file_paths:
            print("    " + path)
        print("")

        file_batch = client.beta.vector_stores.file_batches.upload_and_poll(vector_store_id=vector_store.id, files=file_streams)
        print("[api-state]: uploaded file streams to the vector store")
        vector_store_id = vector_store.id

    assistant = client.beta.assistants.update(assistant_id=assistant.id, tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}})
    print("[api-state]: updated assistant with uploaded vector store")

    thread = client.beta.threads.create(messages=[ {"role": "user", "content": args.prompt } ])
    print("[api-state]: created thread with assistant (no previous history)")

    run = client.beta.threads.runs.create_and_poll(thread_id=thread.id, assistant_id=assistant.id)
    print("[api-state]: created runstate with assistant")

    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
    message_content = messages[0].content[0].text
    annotations = message_content.annotations
    citations = []
    for index, annotation in enumerate(annotations):
        message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
        if file_citation := getattr(annotation, "file_citation", None):
            cited_file = client.files.retrieve(file_citation.file_id)
            citations.append(f"[{index}] {cited_file.filename}")
    print("[api-state]: finished running thread")

    print("")
    print("")

    print("Our Question:")
    print("-------------")
    print(args.prompt)
    print("")
    print("")
    print("Model output:")
    print("-----------")
    print(message_content.value)
    print("")
    print("")
    print("")
    print("")
    print("Model Citations:")
    print("----------------")
    print("\n".join(citations))

if __name__ == "__main__":
    args = parse_arguments()
    main(args)

