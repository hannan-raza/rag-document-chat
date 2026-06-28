import boto3
import json

session = boto3.Session(profile_name="newrag")
sqs = session.client("sqs", region_name="us-east-1")

QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/726959544827/rag-ingestion-queue"

# the message: which PDF, in which S3 bucket, to process
message = {
    "bucket": "rag-learning-hannan-7842",
    "key": "transformer.pdf",
}

response = sqs.send_message(
    QueueUrl=QUEUE_URL,
    MessageBody=json.dumps(message),
)

print("Message sent.")
print(f"  Message ID: {response['MessageId']}")
print(f"  Body: {json.dumps(message)}")