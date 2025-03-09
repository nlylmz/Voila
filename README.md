# VOILA: Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning

[![Paper](https://img.shields.io/badge/Paper-Available-blue)]() 
[![Homepage](https://img.shields.io/badge/Homepage-Visit-green)]() 
[![Huggingface](https://img.shields.io/badge/Huggingface-Model-orange)](https://huggingface.co/datasets/nlylmz/VOILA)

Accepted to ICLR 2025!  

---

Creation requires the highest cognitive skills in the learning process compared to evaluation (Bloom’s taxonomy of educational objectives). However, many current multimodal reasoning tasks rely on multiple-choice formats, where models select a solution from a predefined set. To attain human-level cognitive intelligence, MLLMs must go beyond evaluating options; they must generate solutions for complex tasks that require advanced reasoning skills. In response to this challenge, we introduce VOILA: a large-scale, open-ended, dynamic benchmark of up to 6.4M data designed to evaluate MLLMs’ perceptual understanding and abstract relational reasoning. VOILA employs an analogical mapping approach in the visual domain, requiring models to generate an image that completes an analogy between two given image pairs, reference and application, without relying on predefined choices.

---

[Nilay Yilmaz](https://www.linkedin.com/in/nilay-yilmaz/) | Maitreya Patel | Yiran Lawrence Luo | Tejas Gokhale | Chitta Baral | Suren Jayasuriya | Yezhou Yang 

![voila_data](https://github.com/user-attachments/assets/19f07148-d4d2-4340-9edd-114150aa3f9a)

## 📢 News  
- 🚀 [02/26/2025] The Leaderboard page is coming soon!
- 🚀 [02/26/2025] The using instructions are coming soon! 
- 🚀 [02/26/2025] Our paper is accepted by ICLR 2025!  
- 🚀 [02/25/2025] We upload our VOILA benchmark to Huggingface.  

---

## 💡 Highlights  
- 🔥 **Multiple Atomic Reasoning**: VOILA employs Analogical reasoning which consists of diverse atomic abilities; perceptual understanding, mapping abstract relationships between visual contents, and transferring relational patterns to novel cases.
- 🔥 **Open-ended Multi-Step Evaluation**: Departing from conventional multimodal reasoning tasks rely on multiple-choice formats, where models select a solution from a predefined set, VOILA applies open-ended evaluation with the multi-step approach by comparing the results with ground truth values at each step.
- 🔥 **Dynamic**: Unlike static datasets, VOILA allows the generation of over **6.4M** distinct visual analogy scenarios utilizing manually cleaned 7,280 diverse images across 14 subject types, 13 actions, and 4 numeric values by adjusting flexible property-rule configuration, offering a scalable and adaptable evaluation platform for MLLMs.
- 🔥 **Rule-Based**: VOILA contains analogies with three properties and four rules (Stable, Change, Arithmetic, and Distraction) applied to these properties.
- 🔥 **Two Difficulty Levels**: To introduce varying levels of difficulty, we created two sub-datasets: **VOILA-WD** and **VOILA-ND**. VOILA-WD applies the Distraction rule which requires models to discover and filter out the irrelevant changes among properties while solving analogy questions.
---

##  VOILA Benchmark  

The VOILA benchmark was designed to evaluate the abstract reasoning capabilities of MLLMs. This task challenges models to process perceptual information and apply relational reasoning by interpreting visual content from three given images to generate a fourth image according to a specified pattern. VOILA is a large-scale dataset that dynamically generates visual analogy questions based on demand and configuration. The dataset can generate over 6.4M questions, distributed across 19 unique structures and utilizing a total of 7,280 images which makes VOILA highly scalable and adaptable to various configurations.

### Dataset Creation Pipeline

The figure below illustrates the process of constructing VOILA

![voila_data_pipe (2)](https://github.com/user-attachments/assets/cbe21812-4173-4bb6-a132-a4f05e86790f)


### Multi-step Reasoning and Evaluation Pipeline 

The diagram below illustrates the reasoning and evaluation process of VOILA

![arch_voila](https://github.com/user-attachments/assets/23b13e8b-e330-4d14-bb37-29c80b45f5ce)

The top section illustrates two visual input formats. The left side of the MLLMs connection displays the four primary tasks along with their corresponding prompts, while the right side presents the expected outcomes for each task. The results are scored in the evaluation stage utilizing GPT-4o and ground truths.

## 🔧 Usage 

### Dataset Generation

To dynamically generate your visual analogy questions using generated images, you need to run create_dataset.py which supports creating both training and testing datasets with options for distractions and image collages. You need to download the Train_Images and Test_Images folders to access the images required for generating questions.

Run the script using the following command:
```bash
python create_dataset.py --csv_output <path_to_csv> --dataset <training/testing> --distraction <yes/no> --count <number> --collage <yes/no>
```
#### Arguments: 
```
csv_output (str, required): Path to save the output CSV file.
dataset (str, required): Choose between training or testing datasets.
distraction (str, required): Choose whether to add distractions (yes or no).
count (int, required): Total number of questions to generate. Ensure it is evenly divisible by the number of rules (7 or 19).
collage (str, required): Choose whether to create image collages (yes or no).
```
### Model Evaluation

To evaluate models on VOILA, we provide the inference_test_VOILA_WD.py script, which downloads the VOILA_WD dataset (with distractions) from Hugging Face and performs step-by-step evaluation for the specified model. The model processes the input data, generates outputs, and saves the results in a JSON file.

Run the script using the following command:
```bash
python inference_test_VOILA_WD.py --model_name <huggingface_pretrained_model>
```

#### Arguments: 
```
model_name (str, required): Name of the pretrained model to use.
dataset_name (str, default: nlylmz/VOILA): Name of the Hugging Face dataset to load.
output_path (str, default: results.json): Path to save the output JSON file.
device_map (str, default: auto): Device mapping strategy (auto, cpu, cuda, etc.).
max_new_tokens (int, default: 2048): Maximum number of new tokens to generate.
temperature (float, default: 0.0): Controls randomness in text generation. Higher values produce more diverse outputs.
```
#### Prompts: 
Because of the Distraction rule, the prompt used for testing models at the third step differs between VOILA_WD and VOILA_ND.

##### VOILA_WD:
```
Step 1: "Describe the content of the first three images in one sentence using the count of subjects and actions in the format of 'Image : Description'"
Step 2: "Identify the changed and unchanged properties observed between the first and second images, focusing on count of subjects, subject types, and action properties. For the count of subjects, consider the change in either increase or decrease."
Step 3: "Apply the identified unchanged and changed properties to Image 3 to predict the fourth image. Give me the answer for the fourth image in the format of 'The answer is number = {number}, subject = {subject}, action = {action}'. Use the following rules to determine the properties for the fourth image: 1. If a property remains constant between Image 1 and Image 2, the property in the fourth image will have the same value as the property from Image 3. 2. If a property (excluding number of subjects) changes between Image 1 and Image 2 and is the same in Image 1 and Image 3, set the property value from Image 2 to the fourth image. Otherwise, set it to 'any'. 3. To determine the number of subjects in the fourth image, apply the increase or decrease rate observed from Image 1 to Image 2 to the number of subjects in Image 3. If the result is less than one, set the number property to 'any'."
Step 4: "Generate the image based on the following description {output}."
```
##### VOILA_ND:
```
Step 3: "Apply the identified unchanged and changed properties to Image 3 to predict the fourth image. Give me the answer for the fourth image in the format of 'The answer is number = {number}, subject = {subject}, action = {action}'. Use the following rules to determine the properties for the fourth image: 1. If a property remains constant between Image 1 and Image 2, the property in the fourth image will have the same value as the property from Image 3. 2. If a property (excluding number of subjects) changes between Image 1 and Image 2 and is the same in Image 1 and Image 3, set the property value from Image 2 to the fourth image. 3. To determine the number of subjects in the fourth image, apply the increase or decrease rate observed from Image 1 to Image 2 to the number of subjects in Image 3."
```

### Scoring

To score the results of Multi-Modal Large Language Models (MLLMs) using the GPT-4 Batch API, we provide the score_model_results.py script. This script processes inference results, compares them with ground truth data, generates evaluation outputs in JSONL format, and prints the scores for each step at the terminal. As the script uses Batch API, it might take time to finish the task. To access the ground truth data, you need to download the CSV file, which contains detailed information for each question. 
##### If you change the template of the CSV file, you need to change the row number in the code!

Run the script using the following command:
```bash
python score_model_result.py --json_file_name <path_to_mllm_results> --csv_file_name <path_to_csv> --distraction <yes/no> --open_api_key <your_api_key>
```

#### Arguments: 
```
json_file_name (str, required): Path to the JSON file containing MLLM inference results.
csv_file_name (str, required): Path to the CSV file for evaluation.
distraction (str, required): Choose whether to enable (yes) or disable (no) distractions during evaluation (required).
batch_jsonl_file (str, default: "batch"): Path to save the generated JSONL file for batch processing.
output_batch_job_file(str, default: "results"): Path to save the final evaluation output JSONL file.
open_api_key (str, required): OpenAI API key for accessing GPT-4 Batch API.
```

## 🖋️ Citation  

```
@inproceedings{
yilmaz2025voila,
title={Voila: Evaluation of {MLLM}s For Perceptual Understanding and Analogical Reasoning},
author={Nilay Yilmaz and Maitreya Patel and Yiran Lawrence Luo and Tejas Gokhale and Chitta Baral and Suren Jayasuriya and Yezhou Yang},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=q5MUMlHxpd}
}
```

