import sagemaker
from sagemaker import get_execution_role
import boto3
from time import gmtime, strftime, sleep
from sagemaker.amazon.amazon_estimator import get_image_uri

prefix = 'sagemaker-scikit-learn-2020-06-14-10-02-19-043'

role = get_execution_role()
smclient = boto3.Session().client('sagemaker')
region = boto3.Session().region_name
bucket = sagemaker.Session().default_bucket()

training_image = get_image_uri(region, 'xgboost', '0.90-1')
tuning_job_name = 'xgboost-tuningjob-' + strftime("%d-%H-%M-%S", gmtime())

print(tuning_job_name)

tuning_job_config = {
    "ParameterRanges": {
      "CategoricalParameterRanges": [],
      "ContinuousParameterRanges": [
        {
          "MaxValue": "1",
          "MinValue": "0",
          "Name": "eta",
        },
        {
          "MaxValue": "1",
          "MinValue": "0",
          "Name": "gamma",            
        }
      ],
      "IntegerParameterRanges": [
        {
          "MaxValue": "10",
          "MinValue": "1",
          "Name": "max_depth",
        }
      ]
    },
    "ResourceLimits": {
      "MaxNumberOfTrainingJobs": 20,
      "MaxParallelTrainingJobs": 3
    },
    "Strategy": "Bayesian",
    "HyperParameterTuningJobObjective": {
      "MetricName": "validation:rmse",
      "Type": "Minimize"
    }
  }
     
s3_input_train = 's3://{}/{}/train'.format(bucket, prefix)
s3_input_test ='s3://{}/{}/test'.format(bucket, prefix)
    
training_job_definition = {
    "AlgorithmSpecification": {
      "TrainingImage": training_image,
      "TrainingInputMode": "File"
    },
    "InputDataConfig": [
      {
        "ChannelName": "train",
        "CompressionType": "None",
        "ContentType": "csv",
        "DataSource": {
          "S3DataSource": {
            "S3DataDistributionType": "FullyReplicated",
            "S3DataType": "S3Prefix",
            "S3Uri": s3_input_train
          }
        }
      },
      {
        "ChannelName": "validation",
        "CompressionType": "None",
        "ContentType": "csv",
        "DataSource": {
          "S3DataSource": {
            "S3DataDistributionType": "FullyReplicated",
            "S3DataType": "S3Prefix",
            "S3Uri": s3_input_test
          }
        }
      }
    ],
    "OutputDataConfig": {
      "S3OutputPath": "s3://{}/{}/output".format(bucket,prefix)
    },
    "ResourceConfig": {
      "InstanceCount": 1,
      "InstanceType": "ml.m4.xlarge",
      "VolumeSizeInGB": 10
    },
    "RoleArn": role,
    "StaticHyperParameters": {
      "eval_metric": "rmse",
      "num_round": "500",
      "objective": "reg:squarederror",
      "rate_drop": "0.3",
      "learning_rate": "0.7",
      "subsample": "0.8"
    },
    "StoppingCondition": {
      "MaxRuntimeInSeconds": 3600
    }
}

smclient.create_hyper_parameter_tuning_job(HyperParameterTuningJobName = tuning_job_name,
                                            HyperParameterTuningJobConfig = tuning_job_config,
                                            TrainingJobDefinition = training_job_definition)

smclient.describe_hyper_parameter_tuning_job(
    HyperParameterTuningJobName=tuning_job_name)['HyperParameterTuningJobStatus']