import pull_data as loader
train_df = loader.read_data_set('train')

        
import boto3
session = boto3.Session(
    aws_access_key_id='AKIAI2ZXWJ56DIICLVDQ',
    aws_secret_access_key='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',
)
s3 = session.client('s3')
obj = s3.get_object(Bucket='data622-hw3-kats', Key='train.csv')
obj['Body'].read().decode('utf-8')

import pandas as pd
df = pd.read_csv(obj['Body'].read().decode('utf-8'))

obj.get()['Body'].getvalue()



import boto3
import io
import os

os.environ["AWS_ACCESS_KEY_ID"] = "AKIAI2ZXWJ56DIICLVDQ"
os.environ["AWS_SECRET_ACCESS_KEY"] = "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"

s3 = boto3.client('s3')
obj = s3.get_object(Bucket='data622-hw3-kats', Key='train.csv')
df = pd.read_csv(io.BytesIO(obj['Body'].read()))

import pickle
pickle.dump(df, open('tst.pkl', 'wb'))

s3 = boto3.resource('s3')
boto3.resource('s3').meta.client.upload_file('tst.pkl', 'data622-hw3-kats', 'tst.pkl')

s3.put_object(Body=b'tst.pkl', Bucket='data622-hw3-kats', Key='train.csv')


s3.Object('data622-hw3-kats', 'tst.pkl').put(Body=open('tst.pkl', 'rb'))


import boto3

more_binary_data = b'Here we have some more data'

client = boto3.client('s3')
client.put_object(Body=more_binary_data, Bucket='my_bucket_name', Key='tst')