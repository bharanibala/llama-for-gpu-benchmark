# LLama For GPU Benchmark

This work is based on the Fine-tuning Multi-model LLM for Crisis MMD Repo: https://github.com/deeplearning-lab-csueb/Fine-tune-Multimodal-LLM-for-CrisisMMD

### Step 1: 
Set up the environment using the provided requirements.txt file

### Step 2:
Login to huggingface if not create account by signing up, then get your login key which is required for this project. paste the login key in place of line 754 in crisismmd_llama_pseudo_labeler.py

### Step 3:
Request for using the model from HuggingFace (https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) and make sure you received the access, before executing the code. 

### Inside the code:
1. The possible values for for the task is *"humanitarian"* and *"informative"*

2. The files for the test path can be found in CrisisMMD_Modified. In the original dataset, the actual train file contains the combination of tweets and images resulting in too many duplicate rows. Also, the labels of image and tweet texts mismatch for some rows. those rows are removed and placed in the agreed labels. Again, the agreed label has some duplicates, which are filtered and placed in the image_only and text_only files. We are using these files for the test inference here.

3. we use the dev data for the few shot inference. The code takes the no. of few shot instances mentioned and uses that as examples in the prompt however the performance decreases when compared to zero-shot inference.

4. The use_texts and use_images flags are used to determine the type of the execution leading to three different modalities namely text only, image only and text image which has its own prompt need to be used accordingly based on the task.

5. The provided code and dataset can support only the text modality. Please download the images from the original dataset and make sure to do the required code changes to accomodate the images and text image modalities

## Fine-tuning/Training Llama using LoRA:
The 'train_text_only.py' file contains the code. This needs 'pip install unsloth' to be run in addition to the libraries mentioned in requirments.txt. A new file *'text_only.jsonl'* has been added as the load_dataset code expects json data to be in record orient for some reason while the index orient results only in the index numbers as rows rather than the actual rows. Consumes 21GB of GPU.

## CrisisMMD dataset
Original CrisisMMD dataset: https://crisisnlp.qcri.org/crisismmd 

We use the same benchmark splits introduced by Ofli *et al.* (2020), and only consider tweets where the text and image share the same label for consistency. Additionally, we followed the practice in Mandal *et al.* (2024) and merged a few semantically similar classes to streamline classification. Specifically, *“injured or dead people”* and *“missing or found people”* were consolidated into *“affected individuals”*, while *“vehicle damage”* was grouped under *“infrastructure and utility damage”*.

The class distributions across the train/dev/test splits of the dataset for the Tweet Informativeness and Humanitarian Category tasks are as follows:
### Class distribution for *Informativeness* and *Humanitarian Category* tasks

| Task           | Category              | Text Train | Text Dev | Text Test | Text Total | Image Train | Image Dev | Image Test | Image Total |
|----------------|------------------------|------------|----------|-----------|------------|-------------|-----------|------------|-------------|
| Informative    | Informative            | 5,546      | 1,056    | 1,030     | 7,632      | 6,345       | 1,056     | 1,030      | 8,431       |
|                | Not-informative        | 2,747      | 517      | 504       | 3,768      | 3,256       | 517       | 504        | 4,277       |
|                | **Total**              | 8,293      | 1,573    | 1,534     | 11,400     | 9,601       | 1,573     | 1,534      | 12,708      |
| Humanitarian   | Affected individuals   | 70         | 9        | 9         | 88         | 71          | 9         | 9          | 89          |
|                | Rescue/Volunteering    | 762        | 149      | 126       | 1,037      | 912         | 149       | 126        | 1,187       |
|                | Infrastructure damage  | 496        | 80       | 81        | 657        | 612         | 80        | 81         | 773         |
|                | Other relevant         | 1,192      | 239      | 235       | 1,666      | 1,279       | 239       | 235        | 1,753       |
|                | Not-humanitarian       | 2,743      | 521      | 504       | 3,768      | 3,252       | 521       | 504        | 4,277       |
|                | **Total**              | 5,263      | 998      | 955       | 7,216      | 6,126       | 998       | 955        | 8,079       |

If you want to work on the image datasets. please feel free to download the full original dataset from: https://crisisnlp.qcri.org/crisismmd
