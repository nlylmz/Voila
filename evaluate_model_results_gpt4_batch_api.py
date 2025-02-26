import json
from openai import OpenAI
import pandas as pd
import csv
from IPython.display import Image, display

def load_json(json_file_name):
    with open(json_file_name, "r") as json_file:
        json_data = json.load(json_file)
    return json_data

def load_csv_file(csv_file_name):
    # Load text column from CSV file
    text_column_step1 = []
    text_column_step2 = []
    text_column_step3 = []

    with open(csv_file_name, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            text_column_step1.append(row[8])
            text_column_step2.append(row[11])
            text_column_step3.append(row[7])
    return text_column_step1, text_column_step2, text_column_step3

def create_json_task_step1(text_column, json_data):
    # Creating an array of json tasks
    tasks = []
    for text, entry in zip(text_column, json_data):
        question = entry['query0']['question']
        model_answer = entry['query0']['generated_text'][0]

        prompt = f'''Give me the score by comparing the provided texts to the ground truth text for each image focusing on subject numbers, subject types, and actions. Comparing similar subject types, such as "mice" versus "hamsters," or variations in number, such as "man" versus "men," is acceptable as a correct answer. However, differences in age or gender, such as "woman" versus "senior woman," are not acceptable as a correct answer. If the answer is correct, assign 1 point for each property score, otherwise give 0. 
        Ground truth texts:{text}
        Texts to compare: {model_answer}
        Write the output as JSON object for each image without explanation.'''

        task = {
            "custom_id": f"question{question}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-2024-08-06",
                "temperature": 0.1,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ImageTextMatching",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "integer", "const": question},
                                "image_1": {
                                    "type": "object",
                                    "properties": {
                                        "number": {"type": "integer"},
                                        "subject_type": {"type": "integer"},
                                        "action": {"type": "integer"}
                                    },
                                    "additionalProperties": False,
                                    "required": ["number", "subject_type", "action"]
                                },
                                "image_2": {
                                    "type": "object",
                                    "properties": {
                                        "number": {"type": "integer"},
                                        "subject_type": {"type": "integer"},
                                        "action": {"type": "integer"}
                                    },
                                    "additionalProperties": False,
                                    "required": ["number", "subject_type", "action"]
                                },
                                "image_3": {
                                    "type": "object",
                                    "properties": {
                                        "number": {"type": "integer"},
                                        "subject_type": {"type": "integer"},
                                        "action": {"type": "integer"}
                                    },
                                    "additionalProperties": False,
                                    "required": ["number", "subject_type", "action"]
                                }
                            },
                            "additionalProperties": False,
                            "required": ["question", "image_1", "image_2", "image_3"]
                        },
                        "strict": True,
                    }
                },
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            }
        }

        tasks.append(task)
    return tasks

def create_json_task_step2(text_column, json_data):
    # Creating an array of json tasks
    tasks2 = []
    for text, entry in zip(text_column, json_data):
        question = entry['query1']['question']
        model_answer = entry['query1']['generated_text'][0]

        prompt = f'''Give me the score by comparing the provided texts to the ground truth text to determine if the model correctly identifies the unchanged and changed properties between Image 1 and Image 2. The properties to compare are the number of subjects, the subject type, and the actions. If the answer is correct, assign 1 point for each property, otherwise give 0. 
        Ground truth texts:{text}
        Texts to compare: {model_answer}
        Write the output as JSON object for each image without explanation.'''

        task = {
            "custom_id": f"question{question}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-2024-08-06",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "IdentifyChanges",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "integer", "const": question},
                                "number": {"type": "integer"},
                                "subject_type": {"type": "integer"},
                                "action": {"type": "integer"}
                            },
                            "additionalProperties": False,
                            "required": ["question", "number", "subject_type", "action"]
                        },
                        "strict": True,
                    }
                },
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            }
        }

        tasks2.append(task)
    return tasks2

def create_json_task_step3(text_column, json_data, args):
    # Creating an array of json tasks
    tasks3 = []

    for text, entry in zip(text_column, json_data):
        question = entry['query2']['question']
        model_answer = entry['query2']['generated_text'][0]

        words = text.split()

        if "male" in text or "female" in text or "senior" in text:
            words.insert(3, "action =")
        else:
            words.insert(2, "action =")

        words.insert(1, "subject =")
        result = ' '.join(words)
        text_format = "number = " + result

        if args.distraction == 'no':
            prompt = f'''Give me the score by comparing the provided texts to the ground truth text to determine if the model correctly identifies the properties of subject numbers, subject types, and actions in order. If the answer is correct, assign 1 point for each property, otherwise give 0. 
                                Ground truth texts: {text_format}
                                Texts to compare: {model_answer}
                                Write the output as JSON object for each image without explanation.'''

        elif args.distraction == 'yes':
            prompt = f'''Give me the score by comparing the provided texts to the ground truth text to determine if the model correctly identifies the properties of subject numbers, subject types, and actions in order. If ground truth text contains the value "any" for a property, it only counts as correct if the provided text also has 'any' for that property. Otherwise, the answer is incorrect. If the answer is correct, assign 1 point for each property, otherwise give 0. 
                    Ground truth texts: {text_format}
                    Texts to compare: {model_answer}
                    Write the output as JSON object for each image without explanation.'''

        task = {
            "custom_id": f"question{question}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-2024-08-06",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "FindingLastImage",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "integer", "const": question},
                                "number": {"type": "integer"},
                                "subject_type": {"type": "integer"},
                                "action": {"type": "integer"}
                            },
                            "additionalProperties": False,
                            "required": ["question", "number", "subject_type", "action"]
                        },
                        "strict": True,
                    }
                },
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            }
        }

        tasks3.append(task)
    return tasks3

def create_and_upload_jsonl_file(client, batch_jsonl_file, tasks):
    file_name = batch_jsonl_file
    with open(file_name, 'w') as file:
        for obj in tasks:
            file.write(json.dumps(obj) + '\n')

    # UPLOADING THE FILE
    batch_file = client.files.create(
        file=open(file_name, "rb"),
        purpose="batch"
    )
    return batch_file

def create_batch_job(client, batch_file):
    # CREATING BATCH JOB
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    return batch_job

def wait_for_batch_job_output(client, created_batch_job, poll_interval=60):
    # WAITING BATCH JOB OUTPUT FILE
    start_time = time.time()

    while True:
        # Retrieve the batch job details
        batch_job = client.batches.retrieve(created_batch_job.id)

        # Check if the output_file_id has been created
        if batch_job.output_file_id:
            print(f"Output file created: {batch_job.output_file_id}")
            return batch_job.output_file_id

        # Calculate elapsed time and print a status update
        elapsed_time = time.time() - start_time
        print(f"Still waiting... Elapsed time: {elapsed_time:.2f} seconds ({poll_interval} seconds interval)")

        # Wait before checking again
        time.sleep(poll_interval)

def save_results(client, output_file_id, output_batch_job_file):
    result_file_id = output_file_id
    result = client.files.content(result_file_id).content

    result_file_name = output_batch_job_file

    with open(result_file_name, 'wb') as file:
        file.write(result)

def load_results(output_batch_job_file):
    results = []
    with open(output_batch_job_file, 'r') as file:
        for line in file:
            # Parsing the JSON string into a dict and appending to the list of results
            json_object = json.loads(line.strip())
            results.append(json_object)
    return results

def calculate_accuracy_step1(results):
    totals = {
        'number': 0,
        'subject_type': 0,
        'action': 0,
        'calculated_image_text_matching': 0
    }

    num_questions = len(results)

    for item in results:
        # Extract JSON content
        content_json_str = item['response']['body']['choices'][0]['message']['content']
        content_data = json.loads(content_json_str)

        # Initialize sums for the current item
        sums = {
            'number': 0,
            'subject_type': 0,
            'action': 0,
            'calculated_image_text_matching': 0
        }

        # Aggregate values for each image
        for image_key in ['image_1', 'image_2', 'image_3']:
            image_data = content_data.get(image_key, {})
            sums['number'] += image_data.get('number', 0)
            sums['subject_type'] += image_data.get('subject_type', 0)
            sums['action'] += image_data.get('action', 0)

            # Check if number, subject_type, and action are all 1
            if (image_data.get('number', 0) == 1 and
                    image_data.get('subject_type', 0) == 1 and
                    image_data.get('action', 0) == 1):
                sums['calculated_image_text_matching'] += 1
                # content_data.get('question')

        # Update totals with current item sums
        for key in totals:
            totals[key] += sums[key]

    print("The result of Step 1:")
    print(f"Total sum of numbers: {(totals['number'] * 100 / num_questions / 3)}")
    print(f"Total sum of subject types: {(totals['subject_type'] * 100 / num_questions / 3)}")
    print(f"Total sum of actions: {(totals['action'] * 100 / num_questions / 3)}")
    print(f"Total sum of calculated_image_text_matching: {(totals['calculated_image_text_matching'] * 100 / num_questions / 3)}")

def calculate_accuracy_step2(results):
    totals = {
        'number': 0,
        'subject_type': 0,
        'action': 0,
        'calculated_identify_changes': 0
    }
    num_questions = len(results)
    # Process each result item
    for item in results:
        # Extract JSON content
        content_json_str = item['response']['body']['choices'][0]['message']['content']
        if (content_json_str is not None):
            content_data = json.loads(content_json_str)

        # Initialize sums for the current item
        sums = {
            'number': 0,
            'subject_type': 0,
            'action': 0,
            'calculated_identify_changes': 0
        }

        if (content_data.get('number', 0) == 1):
            sums['number'] += content_data.get('number', 0)
        if (content_data.get('subject_type', 0) == 1):
            sums['subject_type'] += content_data.get('subject_type', 0)
        if (content_data.get('action', 0) == 1):
            sums['action'] += content_data.get('action', 0)

        # Check if number, subject_type, and action are all 1
        if (content_data.get('number', 0) == 1 and
                content_data.get('subject_type', 0) == 1 and
                content_data.get('action', 0) == 1):
            sums['calculated_identify_changes'] += 1
        # Update totals with current item sums
        for key in totals:
            totals[key] += sums[key]

    # Print results
    print(f"Total sum of numbers: {(totals['number'] * 100 / num_questions)}")
    print(f"Total sum of subject types: {(totals['subject_type'] * 100 / num_questions)}")
    print(f"Total sum of actions: {(totals['action'] * 100 / num_questions)}")
    print(f"Total sum of calculated_identify_changes: {(totals['calculated_identify_changes'] * 100 / num_questions)}")

def calculate_accuracy_step3(results):
    totals = {
        'number': 0,
        'subject_type': 0,
        'action': 0,
        'calculated_finding_last_image': 0
    }

    num_questions = len(results)
    correctly_answered_questions = []
    # Process each result item
    for item in results:
        # Extract JSON content
        content_json_str = item['response']['body']['choices'][0]['message']['content']
        if content_json_str is not None:
            content_data = json.loads(content_json_str)

        # Initialize sums for the current item
        sums = {
            'number': 0,
            'subject_type': 0,
            'action': 0,
            'calculated_finding_last_image': 0
        }

        if (content_data.get('number', 0) == 1):
            sums['number'] += content_data.get('number', 0)
        if (content_data.get('subject_type', 0) == 1):
            sums['subject_type'] += content_data.get('subject_type', 0)
        if (content_data.get('action', 0) == 1):
            sums['action'] += content_data.get('action', 0)

        # Check if number, subject_type, and action are all 1
        if (content_data.get('number', 0) == 1 and
                content_data.get('subject_type', 0) == 1 and
                content_data.get('action', 0) == 1):
            sums['calculated_finding_last_image'] += 1
            num_questions = content_data.get('question', 0)
            correctly_answered_questions.append(num_questions)

        # Update totals with current item sums
        for key in totals:
            totals[key] += sums[key]

    # Print results
    print(f"Total sum of numbers: {(totals['number'] * 100 / num_questions)}")
    print(f"Total sum of subject types: {(totals['subject_type'] * 100 / num_questions)}")
    print(f"Total sum of actions: {(totals['action'] * 100 / num_questions)}")
    print(f"Total sum of calculated_finding_last_image: {(totals['calculated_finding_last_image'] * 100 / num_questions)}")

def pipeline_for_steps(client, tasks, batch_jsonl_file, output_batch_job_file):
    batch_file = create_and_upload_jsonl_file(client, batch_jsonl_file, tasks)
    batch_job = create_batch_job(client, batch_file)
    output_file_id = wait_for_batch_job_output(client, batch_job)
    save_results(client, output_file_id, output_batch_job_file)

def main(args):

    client = OpenAI(api_key=args.open_api_key)

    #Load inference model results and ground truths
    json_data = load_json(args.json_file_name)
    csv_data_step1, csv_data_step2, csv_data_step3 = load_csv_file (args.csv_file_name)

    # Create batch job for GPT4-API evaluation
    tasks1 = create_json_task_step1(csv_data_step1, json_data)
    tasks2 = create_json_task_step2(csv_data_step2, json_data)
    tasks3 = create_json_task_step3(csv_data_step3, json_data)

    #Sequence of Working
    # Process each batch of tasks
    #process_batch(client, tasks1, args.batch_jsonl_file + "_step1.jsonl", args.output_batch_job_file + "_step1.jsonl")
    #process_batch(client, tasks2, args.batch_jsonl_file + "_step2.jsonl", args.output_batch_job_file + "_step2.jsonl")
    #process_batch(client, tasks3, args.batch_jsonl_file + "_step3.jsonl", args.output_batch_job_file + "_step3.jsonl")

    ##Parallel Working
    batches = [
        (tasks1, args.batch_jsonl_file + "_step1.jsonl", args.output_batch_job_file + "_step1.jsonl"),
        (tasks2, args.batch_jsonl_file + "_step2.jsonl", args.output_batch_job_file + "_step2.jsonl"),
        (tasks3, args.batch_jsonl_file + "_step3.jsonl", args.output_batch_job_file + "_step3.jsonl"),
    ]


    ##Parallel Working
    # Use ThreadPoolExecutor to process batches concurrently
    with ThreadPoolExecutor() as executor:
        future_to_batch = {
            executor.submit(process_batch, client, tasks, batch_jsonl_file, output_batch_job_file): (
            tasks, output_batch_job_file)
            for tasks, batch_jsonl_file, output_batch_job_file in batches
        }

        for future in as_completed(future_to_batch):
            tasks, output_batch_job_file = future_to_batch[future]
            try:
                future.result()  # This will raise an exception if the function failed
                print(f"Batch processing completed for {output_batch_job_file}.")
            except Exception as e:
                print(f"Batch processing failed for {output_batch_job_file} with exception: {e}")

    # Read results for each step and show accuracy
    results1 = load_results(args.output_batch_job_file + "_step1.jsonl")
    results2 = load_results(args.output_batch_job_file + "_step2.jsonl")
    results3 = load_results(args.output_batch_job_file + "_step3.jsonl")

    calculate_accuracy_step1(results1)
    calculate_accuracy_step2(results2)
    calculate_accuracy_step2(results3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MLLMs results with GPT-4 Batch API")
    parser.add_argument("--json_file_name", type=str, required=True, help="Path to MLLMs inference results of JSON file")
    parser.add_argument("--csv_file_name", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--distraction", type=str, choices=["no", "yes"], required=True, help="Choose the option for distraction: yes, no")
    parser.add_argument("--batch_jsonl_file", type=str, default="batch", help="Path to save the creation of JSONL file")
    parser.add_argument("--output_batch_job_file", type=str, default="results", help="Path to save the output JSONL file")
    parser.add_argument("--open_api_key", type=str, required=True, help="Key for Open API ")

    args = parser.parse_args()
    main(args)
