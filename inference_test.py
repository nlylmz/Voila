import torch
from tqdm import tqdm
import json
import logging
import argparse
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from datasets import load_dataset

def setup_logging():
    logging.basicConfig(
        filename='inference.log',
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def load_processor_and_model(model_name, device):
    logging.info(f"Loading processor and model: {model_name}")
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map=device
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map=device
    )
    return processor, model

def generate_text(inputs, model, processor, generation_config):
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    output = model.generate_from_batch(
        inputs,
        generation_config,
        tokenizer=processor.tokenizer
    )
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text

def main(args):
    setup_logging()
    processor, model = load_processor_and_model(args.model_name, args.device_map)

    # Load the dataset from Hugging Face Hub
    logging.info(f"Loading dataset from Hugging Face Hub: {args.dataset_name}")
    hf_dataset = load_dataset(args.dataset_name, split='train')  # Adjust split if necessary

    json_file_path = args.output_path
    data = []

    if (args.distraction == "no"):
        queries = [
            "Describe the content of the first three images in one sentence using the count of subjects and actions in the format of 'Image : Description'",
            "Identify the changed and unchanged properties observed between the first and second images, focusing on count of subjects, subject types, and action properties. For the count of subjects, consider the change in either increase or decrease.",
            "Apply the identified unchanged and changed properties to the Image 3 to predict the fourth image. Give me the answer for the fourth image in the format of 'The answer is number = {number}, subject = {subject}, action = {action}'. Use the following rules to determine the properties for the fourth image: 1. If a property remains constant between Image 1 and Image 2, the property in the fourth image will have the same value as the property from Image 3. 2. If a property (excluding number of subjects) changes between Image 1 and Image 2 and is the same in Image 1 and Image 3, set the property value from Image 2 to the fourth image. 3. To determine the number of subjects in the fourth image, apply the increase or decrease rate observed from Image 1 to Image 2 to the number of subjects in Image 3."
        ]
    elif (args.distraction == "yes"):
        queries = [
            "Describe the content of the first three images in one sentence using the count of subjects and actions in the format of 'Image : Description'",
            "Identify the changed and unchanged properties observed between the first and second images, focusing on count of subjects, subject types, and action properties. For the count of subjects, consider the change in either increase or decrease.",
            "Apply the identified unchanged and changed properties to the Image 3 to predict the fourth image. Give me the answer for the fourth image in the format of 'The answer is number = {number}, subject = {subject}, action = {action}'. Use the following rules to determine the properties for the fourth image: 1. If a property remains constant between Image 1 and Image 2, the property in the fourth image will have the same value as the property from Image 3. 2. If a property (excluding number of subjects) changes between Image1 and Image 2 and is the same in the Image 1 and Image 3, set the property value from Image 2 to the fourth image. Otherwise, set it to 'any'. 3. To determine the number of subjects in the fourth image, apply the increase or decrease rate observed from Image 1  to Image 2 to the number of subjects in Image 3. If the result is less than one, set the number property to 'any'."
        ]

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        do_sample=True,
        temperature=args.temperature
    )

    for idx in tqdm(range(len(hf_dataset)), desc="Processing dataset"):
        tmp_ = {}
        item = hf_dataset[idx]

        # Load images from the dataset
        images = [item['image1'], item['image2'], item['image3']]
        overall_text = ""

        for kk, query in enumerate(queries):
            overall_text += query + " "
            inputs = processor.process(
                images=images,
                text=overall_text
            )

            generated_text = generate_text(inputs, model, processor, generation_config)
            overall_text += generated_text + " "
            logging.info(f"Generated text for index {idx}, query {kk}: {generated_text}")

            tmp_[f'query{kk}'] = {
                "row_index": idx,
                "query": query,
                "generated_text": generated_text
            }

        data.append(tmp_)

    with open(json_file_path, "w") as json_file:
        json.dump(data, json_file)
    logging.info(f"Results saved to {json_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLLM model inference with Hugging Face dataset")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pretrained model")
    parser.add_argument("--dataset_name", type=str, default='nlylmz/VOILA', help="Name of the Hugging Face dataset to load")
    parser.add_argument("--distraction", type=str, choices=["no", "yes"], required=True, help="Choose the option if the dataset has distraction factor: yes, no")
    parser.add_argument("--output_path", type=str, default="results.json", help="Path to save the output JSON file")
    parser.add_argument("--device_map", type=str, default='auto', help="Device map for model loading")
    parser.add_argument("--max_new_tokens", type=int, default=3048, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for text generation")

    args = parser.parse_args()
    main(args)
