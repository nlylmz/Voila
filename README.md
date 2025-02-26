# VOILA: Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning

[![Paper](https://img.shields.io/badge/Paper-Available-blue)]() 
[![Homepage](https://img.shields.io/badge/Homepage-Visit-green)]() 
[![Huggingface](https://img.shields.io/badge/Huggingface-Model-orange)](https://huggingface.co/datasets/nlylmz/VOILA)

Accepted to ICLR 2025 Main Conference!  

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

### If you change the template of the CSV file, you need to change the row number in the code!

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

