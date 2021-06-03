import pandas as pd
import numpy as np
import os
import functools
import tensorflow as tf
from sklearn.model_selection import train_test_split

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    df = pd.merge(df, ndc_df[['Proprietary Name', 'NDC_Code']], 
                  left_on='ndc_code', right_on='NDC_Code' , how = 'left')
    df['generic_drug_name'] = df['Proprietary Name']
    df = df.drop(['NDC_Code', 'Proprietary Name', 'ndc_code'], axis=1)
    return df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''

    df = df.sort_values('encounter_id')
    first_encounter_values = df.groupby('patient_nbr')[['patient_nbr', 'encounter_id']].head(1)
    first_encounter_df = pd.merge(df, first_encounter_values, on = ['patient_nbr', 'encounter_id'])
        
    return first_encounter_df

# line2encounter
def line2encounter(df, na_values = pd.io.parsers.STR_NA_VALUES, 
                   save_path = 'encounter_df.csv'):    
     
    # convert line to encounter dataframe    
    encounter_df = df.groupby(['patient_nbr', 'encounter_id']).agg(lambda x: 
                                                            list(set([y for y in x if y not in na_values ])) )
    unique_cols = (encounter_df.applymap(lambda x: len(x)).apply(lambda x: x > 1).sum())
    unique_cols = unique_cols[unique_cols == 0].keys()
    encounter_df[unique_cols] = encounter_df[unique_cols].applymap(lambda x: np.nan if len(x)  == 0  else x[0])
    encounter_df = encounter_df.reset_index()
    return(encounter_df)
    

#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    
    # generate ids for train, valid, and test sets
    unique_ids  = df[patient_key].unique()
    np.random.seed(9)
    np.random.shuffle(unique_ids)
    trainids, elseids = train_test_split(unique_ids, test_size = 0.4)
    validids, testids = train_test_split(elseids, test_size = 0.5)
    
    # generate train, valid, and test sets
    train = df[df[patient_key].isin(trainids)]
    validation = df[df[patient_key].isin(validids)]
    test =  df[df[patient_key].isin(testids)]
                                  
    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        tf_categorical_column_with_vocabulary = tf.feature_column.categorical_column_with_vocabulary_file(
            key=c, vocabulary_file = vocab_file_path, num_oov_buckets=1)
        
        tf_indicator_column = tf.feature_column.indicator_column(tf_categorical_column_with_vocabulary)
        
        
        output_tf_list.append(tf_indicator_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature = tf.feature_column.numeric_column(key=col, default_value = default_value, normalizer_fn=normalizer, dtype=tf.float64)
    
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    df['score'] = df[col].apply(lambda x: 1 if x>=5 else 0 )
    student_binary_prediction = df['score'].to_numpy()
    
    return student_binary_prediction
