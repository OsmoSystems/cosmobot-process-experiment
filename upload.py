'''d'''
import os
import boto3

def upload_files(path):
    '''d'''
    session = boto3.Session(
        aws_access_key_id='AKIAICEMNJ4IQ4WTME5Q',
        aws_secret_access_key='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',
        region_name='us-west-2'
    )
    s_3 = session.resource('s3')
    bucket = s_3.Bucket('camera-sensor-experiments')

    print("Uploading Experiment: {}".format(path))

    for subdir, _, files in os.walk(path):
        for file in files:
            full_path = os.path.join(subdir, file)
            with open(full_path, 'rb') as data:
                print("Uploading : {}".format(full_path))
                bucket.put_object(Key=full_path[len(path)+1:], Body=data, ACL='public-read')
