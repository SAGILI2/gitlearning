import boto3
import os


class S3:
    def __init__(self):
        self.s3_bucket_name = "vision-era-uploads"
        # self.bucket_location = boto3.client('s3').get_bucket_location(Bucket=self.s3_bucket_name)
        # key_name = os.path.basename(path)
        # file_name = "media/" + key_name
        # self.s3_resource = boto3.resource('s3')

    def upload_file(self, file_name):
        key_name = os.path.basename(file_name)
        self.s3_resource.meta.client.upload_file(
            Filename=file_name,
            Bucket=self.s3_bucket_name,
            Key=key_name,
            ExtraArgs={"ACL": "public-read"},
        )
        # return "https://s3-{0}.amazonaws.com/{1}".format(self.s3_bucket_name, key_name)
        return "https://s3-{0}.amazonaws.com/{1}/{2}".format(
            self.bucket_location["LocationConstraint"], self.s3_bucket_name, key_name
        )

    def download_from_s3_url(self, s3_url, dst_file):
        s3_url_split = s3_url.split("/")
        bucket_name = s3_url_split[-2]
        key = s3_url_split[-1]
        self.s3_resource.Bucket(bucket_name).download_file(key, dst_file)
