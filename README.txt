# README

This file explains the general rules of accessing, manipulating and storing the files and directories pertaining to the "Multiclass Image Classification Challenge" (MICC) project elaborated by Group 69 for the Deep Learning course in the MsC in Data Science and Advanced Analytics with Specialization in Data Science, TK class 2024/2025. It also details how to obtain the data used to train the model, as well as replicate all of our results. Please take a moment to read it.

# 1. Description

MIC uses a deep convolutional neural network to classify images of animals. The model was trained using 12,000 examples curated from a larger undisclosed database. The data can be obtained by running the ingestion.py script present in this repository. To reproduce our results simply run main.py, after building the environment and running ingestion.py.

# 2. Metadata

1. The images are described in a dataframe containing 7 columns:

    - 'rare_species_id': the id and it's cardinality is the length of the dataset
    - 'eol_content_id': reference to the species in the Encyclopedia of Life (EOL)
    - 'eol_page_id': reference to the species in the Encyclopedia of Life (EOL)
    - kingdom: the kingdom of the species
    - phylum: the phylum of the species ----->  class imbalance (chordata: 83%, arthropoda: 8%, cnidaria: 7%, others: 2%)
    - family: the family of the species ----->  target variable
    - file_path: the path to the image of the species 

# 3. Getting started

## 3.1. For Developers

To get started you may build the environment using the requirements.yml file, on your preferred conda workstation, then run the ingestion.py. and read the versioning rules and guidelines section of this document, then finally check out the TODO section.

## 3.2 For Teachers and others

You may follow the same steps as described before creating the environment using conda, or similar, and running ingestion.py, and then run main.py. Running main allows you to:
	- access the pre-trained models stored in ".pkl"'s in the /pickles directory. 
	- re-train any of the models presented here and then use it.
	- Both retrained models,  the models already present can be used to classify new images - to do this, either provide a single path, a list of paths or a .csv file containing paths, alternatively, it can also ingest URLs, this fails if the resulting object cannot be converted to an image file.
	
# 4. Rules and Guidelines

As this is a small team, authorship and blame are only considered in development stages and not for final models. All contributors are equally responsible for this repository, for that reason, personal forking is heavily discouraged. To facilitate this, the members acknowledge the need for strict adherence to a set of naming conventions and a particular workflow that potentiates them.

## 4.1. Conventions

### 0. article # Pull before Push
Pulling before pushing is a good rule of thumb to avoid merge conflicts. Moreover, as forking is discouraged, to avoid having file conflicts it is forceful that contributors inform others of their current tasks, at their beginning and upon pushing.

### 1 article # Directory Tree
The directory tree is the same for all developers, thus allowing all relative paths to work in a workstation agnostic manner. 

### 2 article # Paths
All paths must be relative, and never point outside of the directory, unless, in those situations where the path provided is in the form of a URL, that points to a web resource - preferably, one for which the URL link is expected to be healthy for the unforeseeable future. 

### 3 article # Data & Other Large Volumes
	# a. As it is heavily discouraged by GitHub all of the data needed for the project must be assigned to the .gitignore file, that is, no data should ever be uploaded to the git.
	# b. The data and all transformations must be obtainable using scripts provided within this repository.
    
### 4 article # Naming conventions
    # a. Naming conventions are not optional.
    # b. notebooks and models inside directories /notebooks and /development are subject to A. "Naming Convention for Code Development". 
    # c. scripts inside directory /scripts are subject to B. "Naming Convention for Utilities".
    # d. model files inside directory /models are subject to C. "Naming Convention for Models"
    
## 4.2 Guidelines for Directories with Specific Needs
This section briefly covers the special guidelines that must be observed in specific directories with special needs. 

## 4.2.1
#### /Models
This directory contains:
    - the current best **performing models*** according to the **Loss Metric**, which follow Naming Convention A (see Naming Conventions below). 
    - and a **/Deprecated** folder, containing both deprecated models and their descriptions.
    - **/Descriptions** which  describes each model using the Document Conventions, Model Description Conventions (see below).
    
* The best performing models with respect to the loss metric.

------------------------------------
------------------------------------

# A Naming Conventions
Naming convention for directories and files should follow ChimeraCase e.g. This_IsChimeraCase.

## 1. Naming Convention for Code Development
Notebook and Model development scripts naming conventions include:
    . **Architecture** - if using a commonly known architecture specify it
    . **Nickname** - a commonly agreed upon shorthand for the model
    . **Version** - the model version
    . **Date** - the date in dd_mm_yyyy format of the release of the version
    . **ForkAuthorInitials** - the name of the person that is working on the fork
    
---   
The model name follows the following schema:
**Example**: `ResNet50_RIZZ_1.0_13_03_2025_JD


## 2. Naming Convention for Utilities
Utilities naming conventions include:
    . **UtilityName** - a purpose specific name for the utility script (should mirror the name of the main class/function inside if applicable)
    . **AuthorInitials** - Initials of the last person to have made changes to the script
    . **Date** - the date in dd_mm_yyyy format of the last time changes were made to the script 
    
---   
The utility name follows the following schema:
**Example**: `DDT_1.0_13_03_2025_RIZZ`

## 3. Naming Convention for Models
Model naming conventions include:
    . **Architecture** - if using a commonly known architecture specify it
    . **Nickname** - a commonly agreed upon shorthand for the model
    . **Version** - the model version
    . **Date** - the date in dd_mm_yyyy format of the release of the version
---   
The model name follows the following schema:
**Example**: `ResNet50_1.0_13_03_2025_RIZZ`
 
This name should be derived from the model description, preferably programatically.

# B Document Conventions
This entry describes the conventions used to fill certain essential documents such as model descriptions

## 1. Model Description Conventions
Model descriptions are .json files that contain all of the information related to the models they pertain to, they have the same name as their /Models.

## 2. Utility Description Conventions
Utilities ought to be signed with a comment line as per the following - # {first name} and {last name}; {dd/mm}, bellow the last signed change to the file, that is, every time a change is made to a utility script, the contributor that last changed it, signs below the last signature as described.

Followed by a brief multi-line comment describing the utility, no further rules are provided on what to actually write, beside it the general concepts of usefulness and conciseness. Keeping a change log is encouraged.

