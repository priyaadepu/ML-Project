## ENd TO END PROJECT
import dagshub
dagshub.init(repo_owner='priyaadepu', repo_name='ML-Project', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
# https://dagshub.com/priyaadepu/ML-Project
# https://github.com/priyaadepu/ML-Project