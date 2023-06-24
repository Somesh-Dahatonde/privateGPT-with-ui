# # #!/usr/bin/env python3
# # from dotenv import load_dotenv
# # from langchain.chains import RetrievalQA
# # from langchain.embeddings import HuggingFaceEmbeddings
# # from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# # from langchain.vectorstores import Chroma
# # from langchain.llms import GPT4All, LlamaCpp
# # import os
# # import argparse
# # import time

# # load_dotenv()

# # embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
# # persist_directory = os.environ.get('PERSIST_DIRECTORY')

# # model_type = os.environ.get('MODEL_TYPE')
# # model_path = os.environ.get('MODEL_PATH')
# # model_n_ctx = os.environ.get('MODEL_N_CTX')
# # model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
# # target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

# # from constants import CHROMA_SETTINGS

# # def main():
# #     # Parse the command line arguments
# #     args = parse_arguments()
# #     embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
# #     db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
# #     retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
# #     # activate/deactivate the streaming StdOut callback for LLMs
# #     callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
# #     # Prepare the LLM
# #     match model_type:
# #         case "LlamaCpp":
# #             llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
# #         case "GPT4All":
# #             llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
# #         case _default:
# #             # raise exception if model_type is not supported
# #             raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
        
# #     qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
# #     # Interactive questions and answers
# #     while True:
# #         query = input("\nEnter a query: ")
# #         if query == "exit":
# #             break
# #         if query.strip() == "":
# #             continue

# #         # Get the answer from the chain
# #         start = time.time()
# #         res = qa(query)
# #         answer, docs = res['result'], [] if args.hide_source else res['source_documents']
# #         end = time.time()

# #         # Print the result
# #         print("\n\n> Question:")
# #         print(query)
# #         print(f"\n> Answer (took {round(end - start, 2)} s.):")
# #         print(answer)

# #         # Print the relevant sources used for the answer
# #         for document in docs:
# #             print("\n> " + document.metadata["source"] + ":")
# #             print(document.page_content)

# # def parse_arguments():
# #     parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
# #                                                  'using the power of LLMs.')
# #     parser.add_argument("--hide-source", "-S", action='store_true',
# #                         help='Use this flag to disable printing of source documents used for answers.')

# #     parser.add_argument("--mute-stream", "-M",
# #                         action='store_true',
# #                         help='Use this flag to disable the streaming StdOut callback for LLMs.')

# #     return parser.parse_args()


# # if __name__ == "__main__":
# #     main()


# from flask import Flask, jsonify, request
# from dotenv import load_dotenv
# from langchain.chains import RetrievalQA
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.vectorstores import Chroma
# from langchain.llms import GPT4All, LlamaCpp
# import os
# import time

# load_dotenv()

# embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
# persist_directory = os.environ.get('PERSIST_DIRECTORY')
# model_type = os.environ.get('MODEL_TYPE')
# model_path = os.environ.get('MODEL_PATH')
# model_n_ctx = os.environ.get('MODEL_N_CTX')
# model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
# target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))
# from constants import CHROMA_SETTINGS

# app = Flask(__name__)

# def initialize_question_answering():
#     embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
#     db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
#     retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
#     callbacks = [StreamingStdOutCallbackHandler()]
#     if model_type == "LlamaCpp":
#         llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
#     elif model_type == "GPT4All":
#         llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
#     else:
#         raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

#     qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
#     return qa

# qa_instance = initialize_question_answering()

# @app.route('/api/question', methods=['POST'])
# def question_api():
#     data = request.get_json()
#     query = data.get('query')
#     response = qa_instance(query)
#     answer = response['result']
#     return jsonify({'answer': answer})

# if __name__ == "__main__":
#     app.run()

import time
from flask import Flask, jsonify, request, Response
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))
from constants import CHROMA_SETTINGS

app = Flask(__name__)

# Lazy initialization of the question-answering model
qa_instance = None

class RealTimeStreamingCallbackHandler(StreamingStdOutCallbackHandler):
    def send_output(self, output: str):
        response = Response(self.stream_output(output), mimetype='text/event-stream')
        response.headers.add('Cache-Control', 'no-cache')
        response.headers.add('Connection', 'keep-alive')
        response.headers.add('X-Accel-Buffering', 'no')
        return response

    def stream_output(self, output: str):
        for char in output:
            yield f"data: {char}\n\n"
            time.sleep(0.1)  # Adjust the delay between characters here

def get_question_answering_model():
    global qa_instance
    if qa_instance is None:
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
        callbacks = [RealTimeStreamingCallbackHandler()]
        if model_type == "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        elif model_type == "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        else:
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

        qa_instance = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    return qa_instance

@app.route('/api', methods=['POST'])
def question_api():
    data = request.get_json()
    query = data.get('query')

    qa_model = get_question_answering_model()
    response = qa_model(query)
    answer = response['result']
    return jsonify({'answer': answer})

if __name__ == "__main__":
    app.run()
