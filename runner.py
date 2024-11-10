import os
import openai
import json
from langsmith.evaluation import LangChainStringEvaluator, evaluate
from langsmith.schemas import Example, Run
from langchain_openai import ChatOpenAI
from langsmith.client import Client

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel

class EvalFormat(BaseModel):
    score : bool
    reason : str




OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

client = Client()
 
def parseJsonFromLLM(json_string:str) -> dict:
    json_string = json_string.replace('\\','').strip()
    obj = dict(json.loads(json_string))
    return obj


def llmCallLangsmith(inputs):
    # client = openai.Client(api_key=OPENAI_API_KEY)
    # chat_ctx = [{
    #             "role": "system",
    #             "content": system_prompt
    #         }]
    # chat_ctx.extend(inputs["chat_history"])
    # chat_ctx.append({"role": "user", "content": inputs["question"]})
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=chat_ctx
    # )
    # llm_answer = response.choices[0].message.content
    return {"llm_answer" : "NA"}

llm_instance = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
guidelines = None

with open('./guidelines.txt', 'r') as f:
    guidelines = f.read()


def eval_helper(answer, question, chat_history):
    client = openai.Client(api_key=OPENAI_API_KEY)
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""
We have a set of guidelines for how to respond to questions \n{guidelines}.
You have to evaluate this step by step.
Step 1 is to understand the question and the guidelines.
Step 2 is to deduce whether the answer is in compliance with the guidelines.
Step 3 is to give a boolean value True or False based on how well the answer adheres to the guidelines, True being compliant and False being non-compliant.

You can be lenient in the case of length and brevity of the answer
Give the reason for your score in the reason field.
                """
            },
            {
                "role": "user",
                "content": f"Question: {question}\Answer: {answer}\nChat History: {chat_history}"
            }
        ],
        response_format=EvalFormat

    )

    response = parseJsonFromLLM(response.choices[0].message.content)
    return response

def premise_evaluator(root_run: Run, example: Example) -> dict:
    llm_answer = root_run.outputs.get('llm_answer')
    ref_answer = example.outputs.get('expected')
    question = example.inputs.get('question')
    chat_history = example.inputs.get('chat_history')
    score_value = eval_helper(ref_answer, question, chat_history)
    return {
        "score": score_value.get('score'),
        "reason": score_value.get('reason'),
        "label": "premise",
        "key": "premise_evaluator"
    }


dataset_name = "chat"

try:
    # Delete existing dataset if it exists
    try:
        existing_ds = client.read_dataset(dataset_name=dataset_name)
        client.delete_dataset(dataset_id=existing_ds.id)
    except:
        pass
        
    # Create new dataset
    ds = client.create_dataset(dataset_name=dataset_name)
    
    data = None
    with open(f'{dataset_name}.json', 'r') as f:
        data = json.load(f)
    _ = client.create_examples(
        inputs=[e["inputs"] for e in data],
        outputs=[e["outputs"] for e in data], 
        dataset_id=ds.id,
    )
except Exception as e:
    print(f"Error creating dataset: {e}")

experiment_results = evaluate(
    lambda inputs: llmCallLangsmith(inputs),
    data=dataset_name,
    evaluators=[premise_evaluator],
    experiment_prefix=f"{dataset_name}-1"
)

# print(experiment_results)