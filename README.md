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
