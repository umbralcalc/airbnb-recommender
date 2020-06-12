import argparse
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

columns_of_interest = ['review_scores_rating','price', 'minimum_nights',
                       'maximum_nights', 'number_of_reviews', 'accommodates', 'guests_included',
                       'bathrooms', 'bedrooms', 'host_total_listings_count', 'host_is_superhost',
                       'host_identity_verified', 'neighbourhood_cleansed',
                       'is_location_exact', 'property_type', 'room_type', 'bed_type',
                       'requires_license', 'instant_bookable', 'cancellation_policy']

# Defined without the label: 'review_scores_rating'
numeric_column_names = ['price', 'minimum_nights',
                        'maximum_nights', 'number_of_reviews', 'accommodates', 'guests_included',
                        'bathrooms', 'bedrooms', 'host_total_listings_count']

categorical_column_names = ['host_is_superhost','host_identity_verified', 'neighbourhood_cleansed',
                            'is_location_exact', 'property_type', 'room_type', 'bed_type',
                            'requires_license', 'instant_bookable', 'cancellation_policy']

def print_shape(df):
    print('Data shape: {}'.format(df.shape))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', type=float, default=0.3)
    args, _ = parser.parse_known_args()
    
    print('Received arguments {}'.format(args))

    input_data_path = os.path.join('/opt/ml/processing/input', 'listings.csv')
    
    print('Reading input data from {}'.format(input_data_path))
    df = pd.read_csv(input_data_path, error_bad_lines=False)
    df = pd.DataFrame(data=df, columns=columns)

    # Reduce to all listings with review scores
    df = df[df['review_scores_rating'].notnull()]

    # Reduce further (only a few more) to all listings with a 'bathrooms' record
    df['bathrooms'] = pd.to_numeric(df['bathrooms'],errors='coerce')
    df['bathrooms'] = df['bathrooms'].notnull().astype(int)

    # Reduce further (only a few more) to all listings with a 'bedrooms' record
    df['bedrooms'] = pd.to_numeric(df['bedrooms'],errors='coerce')
    df['bedrooms'] = df['bedrooms'].notnull().astype(int)

    # Reduce further (only 2 more) to all listings with a 'host_is_superhost' record
    df = df[df['host_is_superhost'].notnull()]

    # Clean price data by removing dollar signs and commas
    df['price'] = df['price'].str.replace('$','').str.replace(',','').astype(float)

    # Convert 'total_listings_count' to more sensible format
    df['host_total_listings_count'] = df['host_total_listings_count'].astype(int)
    
    print('Data shape post-cleaning: {}'.format(df.shape))
    
    split_ratio = args.train_test_split_ratio
    print('Splitting data into train and test sets with ratio {}'.format(split_ratio))
    X_train, X_test, y_train, y_test = train_test_split(df.drop('review_scores_rating', axis=1), \
                                                        df['review_scores_rating'], test_size=split_ratio, random_state=0)

    preprocess = make_column_transformer(
        (numeric_column_names, StandardScaler()),
        (categorical_column_names, OneHotEncoder(handle_unknown='error', sparse=False))
    )
    print('Running preprocessing and feature engineering transformations')
    train_features = preprocess.fit_transform(X_train)
    test_features = preprocess.transform(X_test)
    
    print('Train data shape after preprocessing: {}'.format(train_features.shape))
    print('Test data shape after preprocessing: {}'.format(test_features.shape))
    
    train_features_output_path = os.path.join('/opt/ml/processing/train', 'train_features.csv')
    train_labels_output_path = os.path.join('/opt/ml/processing/train', 'train_labels.csv')
    
    test_features_output_path = os.path.join('/opt/ml/processing/test', 'test_features.csv')
    test_labels_output_path = os.path.join('/opt/ml/processing/test', 'test_labels.csv')
    
    print('Saving training features to {}'.format(train_features_output_path))
    pd.DataFrame(train_features).to_csv(train_features_output_path, header=False, index=False)
    
    print('Saving test features to {}'.format(test_features_output_path))
    pd.DataFrame(test_features).to_csv(test_features_output_path, header=False, index=False)
    
    print('Saving training labels to {}'.format(train_labels_output_path))
    y_train.to_csv(train_labels_output_path, header=False, index=False)
    
    print('Saving test labels to {}'.format(test_labels_output_path))
    y_test.to_csv(test_labels_output_path, header=False, index=False)