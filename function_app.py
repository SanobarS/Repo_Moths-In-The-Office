import logging
import azure.functions as func
import re
import uuid
import numpy as np
from pinecone import Pinecone
from openai import AzureOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


app = func.FunctionApp()
pc = Pinecone(api_key='51f2a1bb-18c5-4349-bacc-16edb5da183d')
index = pc.Index('test-python-embeddings')
completionClient = AzureOpenAI(
        api_key='4331547453b740eb954b45ed37389fc0',
        api_version='2023-03-15-preview',
        azure_endpoint='https://ai-vista2024.openai.azure.com'
    )

embeddingsClient = AzureOpenAI(
    api_key='4331547453b740eb954b45ed37389fc0',
    azure_endpoint='https://ai-vista2024.openai.azure.com',
    api_version='2023-05-15'
)


@app.route(route="get_risk_description", auth_level=func.AuthLevel.FUNCTION)
def get_risk_description(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('get_risk_description function processed a request.')

    risk_statement = req.params.get('risk_statement')
    if not risk_statement:
        return func.HttpResponse("Risk statement missing in the request.", status_code=400)
    
    res_risk_description = risk_description(risk_statement)

    if not res_risk_description:
        return func.HttpResponse(status_code=500)
    else:
        return func.HttpResponse(res_risk_description, status_code=200)


@app.route(route="get_risk_matrices", auth_level=func.AuthLevel.FUNCTION)
def get_risk_matrices(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    risk_statement = req.params.get('risk_statement')
    if not risk_statement:
        return func.HttpResponse("Risk statement missing in the request.", status_code=400)
    
    res_risk_matrices = risk_matrices(risk_statement)

    if not res_risk_matrices:
        return func.HttpResponse(status_code=500)
    else:
        return func.HttpResponse(res_risk_matrices, status_code=200)
    

@app.route(route="get_risk_controls", auth_level=func.AuthLevel.FUNCTION)
def get_risk_controls(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    risk = req.get_json()
    
    if not risk:
        return func.HttpResponse("Risk missing in the request.", status_code=400)
    
    res_risk_controls = risk_controls(risk)

    if not res_risk_controls:
        return func.HttpResponse(status_code=500)
    else:
        return func.HttpResponse(res_risk_controls, status_code=200)
    
@app.blob_trigger(arg_name="myblob", path="moths-in-the-office",
                               connection="031b3f_STORAGE") 
def trigger_risk_analytics(myblob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob"
                f"Name: {myblob.name} ")
    
    content = myblob.read()

    metadata = {
        "filename": myblob.name,
        "url": myblob.uri
    }
    
    process_document(content, metadata)
    

############## HELPERS ##############
def risk_description(risk_statement):
    try:
        prompt = 'You are a risk manager. Provide a risk description based on the given risk_statement and context. Provide description in 100 words and do not include any control measures.'
        prompt += '\nrisk_statement: ' + risk_statement

        context = index.query(
            namespace='ns1',
            vector=get_embedding(prompt)['values'],
            top_k=3,
            include_metadata=True
        )

        chunks = [item['metadata'] for item in context['matches']]
        contextText = [item['chunk'] for item in chunks]

        prompt += '\ncontext: ' + ','.join(contextText)
        prompt += '\nRisk Description: '

        risk_description = get_ai_response(prompt)

        return risk_description
    except Exception as e:
        logging.info(e)
        return
    

def risk_matrices(risk_statement):
    try:
        statement_embeddings = get_embedding(risk_statement)
        statement_embeddings = statement_embeddings['values']
        if pc.describe_index('test-python-embeddings').status['ready']:
            print(index.query(vector=statement_embeddings, top_k=10, include_metadata=True, namespace='ns1'))
            results = index.query(vector=statement_embeddings, top_k=10, include_metadata=True, namespace='ns1')
            prompt = ('You are a risk manager. Provide all possible risk matrices based on the given context. \n'
                    f'risk_statement:{risk_statement}\n'
                    'Json format for risk matrices to be returned: {'
                    '                                                  "Risk_Title": "Risk_Title",'
                    '                                                  "Likelihood": "Range 1(LOW)-5(HIGH)",'
                    '                                                  "Impact": "Range 1(LOW)-5(HIGH)",'
                    '                                                  "Control_Measure": "JSON array of control Suggestions in key value pair of control_title and control_description"'
                    '                                              }. ')

            prompt += '\ncontext: '
            for match in results['matches']:
                prompt += '\n' + match['metadata']['chunk']

            prompt += '\nDo not include risk matrices for which Risk_Title is not relevant to the given risk_statement'

            risk_matrices = get_ai_response(prompt)

            return risk_matrices
        else:
            return
    except Exception as e:
        logging.info(e)
        return


def risk_controls(risk):
    try:
        prompt = ('You are a risk manager. Provide 3 possible control measures for the given risk '
                'based on the given risk_statement, risk_description and risk_title.'
                'Do not include provided risk_controls to the new suggestions.')
        prompt += '\nrisk_statement: ' + risk["risk_statement"]
        prompt += '\nrisk_description: ' + risk["risk_description"]
        prompt += '\nrisk_title: ' + risk["risk_title"]
        prompt += '\nrisk_controls: ' + ", ".join(risk["risk_controls"]) + "."

        prompt += '\nJSON array of control Suggestions in key value pair of control_title and control_description: '  

        risk_controls = get_ai_response(prompt)

        return risk_controls
    except Exception as e:
        logging.info(e)
        return


def get_embedding(text, metadata = {}, model="text-embedding-ada-002"):
    # Replace consecutive spaces, newlines and tabs
    text = re.sub(r'\s+', ' ', text)

    embeddings = embeddingsClient.embeddings.create(input=[text], model=model)

    metadata = { "chunk": text } | metadata
    logging.info(metadata)

    data = {
        "id": str(uuid.uuid4()),
        "values": embeddings.data[0].embedding,
        "metadata": metadata
    }
    return data


def get_ai_response(prompt, model='gpt-4o'):
        response = completionClient.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response.choices[0].message.content

def process_document(content, metadata):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents([Document(page_content=content)])
        vectors = np.array([get_embedding(doc.page_content, metadata) for doc in docs]).tolist()
        index.upsert(vectors=vectors, namespace="ns1")
        logging.info('Embeddings Synced with Pinecone')

        #Add logic to suggest risk statements
    except Exception as e:
        logging.info(e)