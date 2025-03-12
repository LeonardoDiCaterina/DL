# metadata exploration
1. the dataset contains 7 columns:

- 'rare_species_id': the id and it's cardinality is the length of the dataset
- 'eol_content_id': reference to the species in the Encyclopedia of Life (EOL) -----> maybe good for image quality inference (?)
- 'eol_page_id': reference to the species in the Encyclopedia of Life (EOL) -----> maybe good for image quality inference (?)
- kingdom: the kingdom of the species -----> useless the kingdom is the same for all the species (animalia)
- phylum: the phylum of the species -----> HUGE class imbalance (chordata: 83%, arthropoda: 8%, cnidaria: 7%, others: 2%)
- family: the family of the species -----> it's the target variable
- file_path: the path to the image of the species

2. the images are not of the same size -> done function change them dynamically *has to be tested!!!*

# short term goal
- binary classification (cordata- not cordata)

# medium term goal
- multiple classification (phylum)

# end goal 
- multiple classification (family) 
![Dom Toretto](https://w7.pngwing.com/pngs/92/388/png-transparent-vin-diesel-furious-7-dominic-toretto-brian-o-conner-youtube-vin-diesel.png)




# CNN model Naming Convention

---

## **1. Include Model Type**
Specify the type of CNN architecture used (e.g., `ResNet`, `VGG`, `CustomCNN`).

**Example**: `ResNet50_AuthorName_v1`

---

## **2. Author's Name or Initials**
Include the author's name or initials to identify who developed or modified the model.

**Example**: `VGG16_DToretto_v2` (made by Dominic Toretto)

---

## **3. Use Version Numbers**
Adopt a consistent versioning system (e.g., `v1`, `v2`, etc.) to track iterations and improvements.

**Example**: `CustomCNN_WSmith_v3` (third version by Will Smith)

---

## **4. Key Features or Modifications**
Highlight any significant changes or features in the name, such as hyperparameter tuning or regularization techniques.

**Example**: `ResNet50_CNorris_L2Reg_v4` (L2 Regularization of version 4 of ResNet50 by Chuck Norris)

---

## **5. Timestamps for Experiment Logs**
For models generated during experiments, append a timestamp to distinguish between runs on the same day.

**Example**: `VGG16_SDoo_v1_0312` (generated on March 12 by Scooby Doo)

---

## **6. (optional) Give a nickname to the model**
If the model is used frequently, give it a nickname to make it easier to reference.

**Example**: `ResNet50_HMoleman_v1_0312_NiceGlasses` (Hans Moleman's ResNet50 model with a timestamp and nickname)

---

