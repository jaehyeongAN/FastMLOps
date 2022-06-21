import os
import time
import mlflow

PROJECT_ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT_PATH, '_output')


class ModelManager:
    '''
    - set up experiment
    - autologing
    - 
    '''
    def __init__(self, mlflow_client, redis_client, modelwrapper):
        self.mlflow = mlflow
        self.mlflow_client = mlflow_client
        self.redisai_client = redisai_client
        self.modelwrapper = modelwrapper


    def mlflow_setup_experiment(self, expr_name, artifact_location):
        # Set experiment - remote backend store & artifacts store
        if mlflow.get_experiment_by_name(expr_name) is None:
            mlflow.create_experiment(expr_name, artifact_location=artifact_location)
        mlflow.set_experiment(expr_name)


    def mlflow_autologging(self):
        self.mlflow.autolog()


    def mlflow_model_cycling(self, target_expr, train_data, train_labels, eval_data, eval_labels):
        with self.mlflow.start_run() as run:
            run_id = run.info.run_id

            # Training Model
            s = time.time()
            model_wrapper = ModelWrapper(args=None)
            model_wrapper.train(train_data, train_df['intent'])
            print('model train time: ', time.time()-s)

            # Evaluate Model
            model = model_wrapper.model

            if eval_data and eval_labels:
                self.mlflow.sklearn.eval_and_log_metrics(model=model, X=eval_data, y_true=eval_labels, prefix='eval_')
            else:
                self.mlflow.sklearn.eval_and_log_metrics(model=model, X=train_data, y_true=train_labels, prefix='train_')

            # Saving model to ONNX
            s = time.time()
            dummy = train_data[0].reshape(1, -1).astype(np.float32)
            model_name = f"{target_expr}-{run_id}"
            if os.path.exists(f"{MODEL_OUTPUT_PATH}") == False:
                os.makedirs(f"{MODEL_OUTPUT_PATH}")
            save_sklearn(model, f'{MODEL_OUTPUT_PATH}/{model_name}.onnx', prototype=dummy)
            print('Saving ONNX model: ',time.time()-s)

            # Uploading model to artifact store 
            model = load_model(f'{MODEL_OUTPUT_PATH}/{model_name}.onnx')
            self.mlflow.onnx.log_model(model, artifact_path='model', registered_model_name=target_expr, conda_env='conda.yaml')

            latest_model = self.mlflow_client.get_latest_versions(target_expr, stages=['None'])[0]
            model_save_path = f"{MODEL_OUTPUT_PATH}/{latest_model.run_id}"
            tag = f"v.{latest_model.version}"

        return {'run_id':run_id, 'model_name':model_name, 'artifact_location':model_save_path, 'model_version':tag}


    def set_mlflow_lateset_model_to_resdisai(self, target_expr):
        latest_model = self.mlflow_client.get_latest_versions(target_expr, stages=['None'])[0]
        model_save_path = f"{MODEL_OUTPUT_PATH}/{latest_model.run_id}"
        tag = f"v.{latest_model.version}"

        # Check if the latest model is on RedisAI
        try:
            model_meta = self.redis_client.modelget(f'{model_name}', meta_only=True)
            print(model_meta)
            print(f">>> Already '{model_name}' is activing in RedisAI🟢")
        except:
            model_meta = None

        # If latest model not exists on RedisAI, Download latest model from artifacts store and Set to RedisAI
        if model_meta == None:
            # Download artifacts from GCS
            if os.path.exists(model_save_path) == False:
                os.makedirs(model_save_path)
            self.mlflow_client.download_artifacts(run_id, "model", model_save_path)

            # Set to RedisAI
            model = load_model(os.path.join(model_save_path, "model/model.onnx"))
            result = self.redisai_client.modelstore(model_name, 'onnx', device, model, tag=tag)
            if result == 'OK':
                print(f">>> {result}, Now model is activing on RedisAI 🟢")
            else:
                print(f">>> {result}, Sync is failed 🔴")

        # remove output dir
        rm_dir(MODEL_OUTPUT_PATH)

        retrun