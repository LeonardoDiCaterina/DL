# Leonardo 13/03 v1


import pandas as pd
import numpy as np
from keras.utils import to_categorical




def create_output_space(df):
    """
    Create the output space for the model. The output space is a dictionary with the following structure:

    Args:
        df (pd.DataFrame): metadata dataframe with the following columns: 'phylum', 'family', 'file_path'

    Returns:
        df (pd.DataFrame): dataframe with the target values for 'phylum' and 'family'
    """
    
    
    u_phylum = df['phylum'].unique()
    u_family = df['family'].unique()
    
    n_classes_phylum = len(u_phylum)
    n_classes_species = len(u_family)
    
    phylum_dict = {u_phylum[i]:i for i in range(n_classes_phylum)}
    family_dict = {u_family[i]:i for i in range(n_classes_species)}
    
    y_phylum = df['phylum'].map(phylum_dict)
    y_family = df['family'].map(family_dict)
    
    y_phylum = to_categorical(y_phylum, num_classes=n_classes_phylum)
    y_family = to_categorical(y_family, num_classes=n_classes_species)
    
    dict_data = {}
    for index,id in enumerate(df['rare_species_id']):
        dict_data[index] = {'phylum':y_phylum[index], 'family':y_family[index], 'file_path':df['file_path'][index]}
    
    
    df_dict = pd.DataFrame.from_dict(dict_data, orient='index')    
    
    return df_dict


def create_is_Chordata_output_space(df):  
    """
    create the target space for the Chordata classification

    Args:
        df (pd.DataFrame): metadata dataframe with the following columns: 'phylum'

    Returns:
        y_Chordata (pd.Series): binary series with 1 if the phylum is Chordata or 0 otherwise
    """
    
      
    y_Chordata = (df['phylum'] == 'Chordata').astype(int)
        
    return y_Chordata