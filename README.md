# Code-Mixed-Machine-Translation
 
## Objective:
The objective of this project is to enhancing machine translation for code-mixed language text by imporving the embedding representation and model architecture.

## Abstract
Machine translation is a rapidly evolving field in Natural Language Processing (NLP) with applications ranging from improving cross-lingual communication to assisting global business operations. In this project, we address the complex challenge of translating code-mixed language text. Our aim is to develop a robust system that can accurately translate mixed language text (Spanglish and Hinglish) into English, recognizing the nuances of language switching within a sentence. With bilingualism being a significant aspect of many team members’ lives, this project has a personal and practical relevance that extends beyond the research realm. Moreover, this work serves as a proof of concept, potentially paving the way for enhanced mixed language translation in various applications.

## Folder Structure.
```
Code-Mixed-Machine-Translation Repo.

|- data
    |- raw (this folder holds raw/downloaded data, test, validation and train data. Note: Do not push the data to github. Instead add a README.md here about how to get data.)
    |- processed (this folder holds processed data, test, validation and train data. Note: Do not push the data to github. Instead add a README.md here about how to get here.)
|- docs (documentation such as proposals and reports fall here)
|- logs (any logging that takes places in this end-to-end pipeline goes here)
|- runs (pretrained models, saved/logged models goes here)
|- nbs (contains jupyter notebook that demonstrates various activities)
|- src (main body of our the codebase)
    |- data (activities related to data such as, data cleaning, pre-processing, EDA, etc falls into this folder.)
    |- machine_translation (defines end-to-end ML pipeline for code-mixed machine translation)
|- requirements.txt (pip requirements (with version) that must be installed prior to running this pipeline)
```

## Tasks
- [X]  Data acquisition
    - You can get the data from the [google drive](https://drive.google.com/drive/folders/1upcPGjw2WwTWYEsSx67IlLWHlvNFJ3Oo?usp=sharing).
    - Once downloaded, place these files inside `./data/raw`. Alternatively, you may change this by setting the `data.base_dir` and `data.dataset-name.dataset.<split>` attributes in the `config.json` file.
- [X]  Data annotations/preparation
    - [X]  Data annotations (when using multiple sources, not applicable for now)
    - [X]  Data cleaning
    - [X]  Data pre-processing
    - [X]  EDA
    - [X]  Feature engineering
- [X]  Machine translation pipeline
    - [X]  Define custom dataset and dataloaders
    - [X]  Model architecture definition
    - [X]  Training/Validation/Testing script
    - [X]  Report/Result geenration script

## Environment Variables
Any environment variables used in this project must be mentioned here.
```
TOKENIZERS_PARALLELISM=False
GOOGLE_APPLICATION_CREDENTIALS=[Optional]
```

## References
* Mentioned in report.

## Notes
* **As much as possible, try to stick to this template. Any improvement/suggestion to this organization is always welcome.**
* **Let us try to write a clean code. Comment where ever necessary so that others can understand and pickup easily.**
* **Jupyter notebooks must contain minimal code. What it means is, if you have a big function, the implementation falls into the appropriate `src` folder and this funciton gets called in the notebook. This stands for classes as well.**
* **If there is anything that you feel is important, please mention them under the appropriate `Tasks` subheadings.**
* **Any tasks that you take, see to it that you are not in the `main` branch. Checkout the `main` branch and start from there. Once done, create a pull request.**
* **To maintain consistency, the naming of all branches except `main` must be done as follows: `[ticket-number] - branch-name`. For example, for the first commit it could have been `[1] - inital-commit`. The commit message can be `[branch-name] - comment`. The ticket number can be found in github projects that you have been added to. Everytime, you create an issue and associate it with a project, a ticket number gets created, please use that. This is not a necessity, but let us try to stick to this format aka professionalism :-p.**