import logging
import azure.functions as func
import re
import os
import json
import uuid
import numpy as np
from pinecone import Pinecone
from openai import AzureOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from azure.storage.blob import BlobServiceClient



app = func.FunctionApp()
azure_storage_connection_string = os.environ["031b3f_STORAGE"]
pc = Pinecone(api_key='51f2a1bb-18c5-4349-bacc-16edb5da183d')
index = pc.Index('test-python-embeddings')
completionClient = AzureOpenAI(
        api_key='4331547453b740eb954b45ed37389fc0',
        api_version='2023-03-15-preview',
        azure_endpoint='https://ai-vista2024.openai.azure.com'
    )

imageClient = AzureOpenAI(
        api_key='4331547453b740eb954b45ed37389fc0',
        api_version='2024-02-01',
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
        return func.HttpResponse(res_risk_description, mimetype="application/json", status_code=200)


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
        return func.HttpResponse(res_risk_matrices, mimetype="application/json", status_code=200)
    

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
        return func.HttpResponse(res_risk_controls, mimetype="application/json", status_code=200)
    

@app.route(route="upload_document", auth_level=func.AuthLevel.FUNCTION)
def upload_document(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    if 'file' not in req.files:
        return {'error': 'No file part in the request'}, 400
    file = req.files['file']
    
    if file.filename == '':
        return {'error': 'No file selected for uploading'}, 400
    
    metadata = upload_document(file)

    if metadata:
        risk_statements = process_document(metadata["content"], metadata, True)
        return func.HttpResponse(risk_statements, mimetype="application/json", status_code=200)
    else:
        return func.HttpResponse(status_code=500)

    
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
    
    process_document(content, metadata, False)

@app.route(route="get_ai_risk_suggestions", auth_level=func.AuthLevel.FUNCTION)
def get_ai_risk_suggestions(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    res_ai_risk_suggestions = ai_risk_suggestions()
    
    if res_ai_risk_suggestions:
        return func.HttpResponse(json.dumps(res_ai_risk_suggestions), mimetype="application/json", status_code=200)
    else:
        return func.HttpResponse(status_code=500)

@app.route(route="generate_heatmap_image", auth_level=func.AuthLevel.FUNCTION)
def generate_heatmap_image(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    risks = req.get_json()
    
    if not risks:
        return func.HttpResponse("Risk missing in the request.", status_code=400)
    
    heatmap_image_url = report_html(risks)

    if heatmap_image_url:
        return func.HttpResponse(heatmap_image_url, status_code=200)
    else:
        return func.HttpResponse(status_code=500)
    
@app.route(route="save_to_risk_register", auth_level=func.AuthLevel.FUNCTION)
def save_to_risk_register(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    risk = req.get_json()

    if not risk:
        return func.HttpResponse("Risk missing in the request.", status_code=400)

    risk["id"] = str(uuid.uuid4())   

    if func_save_to_risk_register(risk):
        metadata = {
            "id": risk["id"]
        }
        risk_statement_embeddings = get_embedding(risk["risk_statement"], metadata)
        risk_description_embeddings = get_embedding(risk["risk_description"], metadata)

        vectors = [risk_statement_embeddings, risk_description_embeddings]

        index.upsert(vectors=vectors, namespace="risk-register")

        return func.HttpResponse(status_code=200)
    else:
        return func.HttpResponse(status_code=500)
    

@app.route(route="suggestions_from_risk_register", auth_level=func.AuthLevel.FUNCTION)
def suggestions_from_risk_register(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    risk_statement = req.params.get('risk_statement')

    risk_suggestions = func_suggestions_from_risk_register(risk_statement)

    return func.HttpResponse(json.dumps(risk_suggestions), mimetype="application/json", status_code=200)


############## HELPERS ##############
def risk_description(risk_statement):
    try:        
        prompt = 'You are a risk manager. Provide a risk_matrix based on the given risk_statement and context.'
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
        prompt += ('Json format for risk_matrix to be returned:{"risk": [{'
'												   "risk_description": "Risk description in 100 words.",'
'                                                  "likelihood": "Range 1(LOW)-5(HIGH)",'
'                                                  "impact": "Range 1(LOW)-5(HIGH)",'
'                                                  "control_measures": "JSON array of 3 most effective control Suggestions in key value pair of control_title and control_description"'
'                                              }]}. ')

        risk_description = get_structured_ai_response(prompt)

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
                    #'Domain: Healthcare and legal\n'
                    'Json format for risk matrices to be returned:{"risks": [{'
                    '                                                  "risk_title": "risk_title",'
                    '                                                  "risk_sector": "risk_sector",'
                    '                                                  "likelihood": "Range 1(LOW)-5(HIGH)",'
                    '                                                  "impact": "Range 1(LOW)-5(HIGH)",'
                    '                                                  "control_measures": "JSON array of control Suggestions in key value pair of control_title and control_description"'
                    '                                              }]}. ')

            prompt += '\ncontext: '
            for match in results['matches']:
                prompt += '\n' + match['metadata']['chunk']

            prompt += '\nDo not include risk matrices for which Risk_Title is not relevant to the given risk_statement.'
            prompt += 'JSON: '

            risk_matrices = get_structured_ai_response(prompt)

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

        prompt += '\nJSON array of control_measures in key value pair of control_title and control_description: '  

        risk_controls = get_structured_ai_response(prompt)

        return risk_controls
    except Exception as e:
        logging.info(e)
        return


def get_embedding(text, metadata = {}, model="text-embedding-ada-002"):
    # Replace consecutive spaces, newlines and tabs
    text = re.sub(r'\s+', ' ', text)

    embeddings = embeddingsClient.embeddings.create(input=[text], model=model)

    metadata = { "chunk": text } | metadata

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


def get_structured_ai_response(prompt, model='gpt-4o'):
    try:
        response = completionClient.beta.chat.completions.parse(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            response_format= { "type": "json_object" }
        )

        return response.choices[0].message.content
    except Exception as e:
        logging.info(e)

def get_image_ai_response(prompt, model='Dalle3'):
    try:
        result = imageClient.images.generate(
                    model=model,
                    prompt=prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1
                )
        
        logging.info(result)

        return json.loads(result.model_dump_json())['data'][0]['url']

    except Exception as e:
        logging.info(e)

def upload_document(file: func.HttpRequest.files):
    try:
        container_name = 'moths-in-the-office'
        blob_name = file.filename

        blob_service_client = BlobServiceClient.from_connection_string(azure_storage_connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)

        blob_client.upload_blob(file.stream, overwrite=True)

        metadata = {
            "filename": blob_name,
            "url": blob_client.url,
            "content": blob_client.download_blob().readall().decode('utf-8')
        }

        return metadata
    except Exception as e:
        logging.info(e)
        return None

def get_risk_blob_client(blob_name = "possible-risk-statements.json"):
    container_name = "risk-store"

    blob_service_client = BlobServiceClient.from_connection_string(azure_storage_connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)

    return blob_client
    

def process_document(content, metadata, getSuggestionResponse=True):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents([Document(page_content=content)])
        vectors = np.array([get_embedding(doc.page_content, metadata) for doc in docs]).tolist()
        index.upsert(vectors=vectors, namespace="ns1")
        logging.info('Embeddings Synced with Pinecone')

        if isinstance(content, bytes):
            content = content.decode('utf-8')

        prompt = 'You are a risk manager. Analyse the CONTENT_DETAILS and provide JSON array of possible risk statements (if any).'
        prompt += '\nResult should be sorted as per the priority of the risks.'
        prompt += '\nResponse Format: {"risks": list[str]}'
        prompt += '\nCONTENT_DETAILS: '
        prompt += '\n' + content

        risk_statements = get_structured_ai_response(prompt)

        if getSuggestionResponse:
            return risk_statements

        blob_client = get_risk_blob_client()
        blob_data = blob_client.download_blob().readall().decode('utf-8')

        result_data = metadata | json.loads(risk_statements)
        result_data = [result_data]

        if len(blob_data) > 0:
            blob_data = json.loads(blob_data)
            result_data = blob_data + result_data

        blob_client.upload_blob(json.dumps(result_data), overwrite=True)
        logging.info('possible-risk-statements updated')
    except Exception as e:
        logging.info(e)


def ai_risk_suggestions():
    try:
        blob_client = get_risk_blob_client()
        blob_data = blob_client.download_blob().readall().decode('utf-8')

        return json.loads(blob_data)
    except Exception as e:
        logging.info(e)


def heatmap_image(risks):
    try:
        prompt = ('I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS:'
                  'Based on the provided risk_Details, Create a risk heatmap matrix with a 5x5 grid. '
                  'The x-axis represents "Impact" (from 1 to 5), and the y-axis represents "Likelihood" '
                  '(from 1 to 5). Use a color gradient ranging from green (low risk) to red (high risk). '
                  'Highlight certain cells based on specific risk values provided, and include a title '
                  '"Risk Heatmap Matrix." Ensure both axes are labeled, and add a color bar to indicate '
                  'the risk count in each cell. Make it visually clear and easy to interpret.')
        prompt += '\nrisk_Details: ' + json.dumps(risks)
        logging.info(prompt)

        image_url = get_image_ai_response(prompt)

        return image_url
    except Exception as e:
        logging.info(e)


def report_html(risk):
    try:
        prompt = ('You are a risk manager. Generate a risk report in html format that can be easily understood '
                  'and professional in nature. Include risk_statement and risk_description at the top of the '
                  'document. Then based on the risk_matrices, explain each Risk_Title and Control_Measure with '
                  'a small description. Along with descriptions, generate a risk matrix of 5X5 for Impact and '
                  'Likelihood for each risk with color code. Highlight the cell having the value (Impact x Likelihood) for the given risk in the matrix. '
                  )
        prompt += '\nrisk_statement: ' + risk["risk_statement"]
        prompt += '\nrisk_description: ' + risk["risk_description"]
        prompt += '\nrisk_Details: ' + json.dumps(risk['risks'])
        logging.info(prompt)

        content = get_ai_response(prompt)

        return content
    except Exception as e:
        logging.info(e)

def func_save_to_risk_register(risk):
    try:
        blob_client = get_risk_blob_client("risk_register.json")
        blob_data = blob_client.download_blob().readall().decode('utf-8')

        risk = [risk]

        if len(blob_data) > 0:
            blob_data = json.loads(blob_data)
            risk = blob_data + risk

        blob_client.upload_blob(json.dumps(risk), overwrite=True)
        logging.info('risk_register updated')
        return True
    except Exception as e:
        logging.info(e)
        return False
    
def func_suggestions_from_risk_register(risk_statement):
    try:
        prompt = 'You are a risk manager. Based on the given risk_statement, find the most relavent risk. Do not return anything if there is no match.'
        prompt += '\nrisk_statement: ' + risk_statement

        context = index.query(
            namespace='risk-register',
            vector=get_embedding(prompt)['values'],
            top_k=3,
            include_metadata=True
        )

        chunks = [item["metadata"] for item in context['matches']]
        blob_client = get_risk_blob_client("risk_register.json")
        blob_data = blob_client.download_blob().readall().decode('utf-8')

        risk_matches = []
        if len(blob_data) > 0:
            risk_data = json.loads(blob_data)
            for chunk in chunks:
                duplicate_match = [risk for risk in risk_matches if risk["id"] == chunk['id']]

                if not duplicate_match:
                    relevance_prompt = "risk_statement:" + risk_statement
                    relevance_prompt += "\nsuggested_risk_statement:" + chunk['chunk']
                    relevance_prompt += "\nIs the risk_statement and suggested_risk_statement related? Rank their relevance on a scale of 1-100%."
                    relevance_prompt += "\nReturn result in json format: { 'is_related': 'true/false', 'relevance_score': '1-100' }"

                    relevance = get_structured_ai_response(relevance_prompt)
                    relevance = json.loads(relevance)

                    if relevance['is_related'].lower() == "true":
                        risk_match = [risk for risk in risk_data if risk["id"] == chunk['id']]
                        if len(risk_match) > 0:
                            risk_match = risk_match[0]
                            risk_match["relevance_score"] = relevance["relevance_score"]
                            risk_matches.append(risk_match)

        return risk_matches
    except Exception as e:
        logging.info(e)  


def generate_html_report():
    try:
        return
    except Exception as e:
        logging.info(e) 