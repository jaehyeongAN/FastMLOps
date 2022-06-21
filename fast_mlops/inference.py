import time


class Inferencer:

    def __init__(self, mlflow_client, redisai_client):
        self.mlflow_client = mlflow_client
        self.redisai_client = redisai_client


    def get_latest_model(self, target_expr):
        ''' 
        Get latest run_id from MLflow 
        '''
        experiment = self.mlflow_client.get_experiment_by_name(target_expr)
        latest_model = self.mlflow_client.get_latest_versions(target_expr)[0]
        print(f"======== Latest Model Version of '{target_expr}' ========")
        print(f"run_id : {latest_model.run_id}")
        print(f"version : {latest_model.version}")
        print(f"source : {latest_model.source}")
        print(f"creation_timestamp : {latest_model.creation_timestamp}")
        print(f"last_updated_timestamp : {latest_model.last_updated_timestamp}")
        print(f'==========================================================')

        model_name = f'{target_expr}-{latest_model.run_id}'

        return model_name


    def redisai_modelexecute(self, model_name, input_tensor):
        ''' 
        input_tensor is registed to RedisAI and used as input of ML model 
        '''
        # tensorset
        self.redisai_client.tensorset(f'{model_name}:in', input_tensor)

        # predict
        self.redisai_client.modelexecute(model_name, inputs=[f'{model_name}:in'], outputs=[f'{model_name}:out1', '_'])


    def inference(self, target_expr, input_tensor):
        ''' 
        Set input_tensor to RedisAI -> Inference  -> Return the result 
        '''
        print(f">> START INFERENCE '{target_expr}' EXPRIMENT'S MODEL ..")

        s = time.time()
        try:
            # Check latest model 
            model_name = self.get_latest_model(target_expr)

            try:
                # Check latest model from RedisAI
                model_meta = self.redisai_client.modelget(model_name, meta_only=True)

                # predict by redisai
                self.redisai_modelexecute(model_name, input_tensor)

                # result
                pred = self.redisai_client.tensorget(f'{model_name}:out1')[0]
                print('---------------------------------------')
                print('INPUT QUERY : ', input_query)
                print('PREDICTED INTENT : ', pred)
                print('---------------------------------------')
                print('processing time : ', time.time()-s)
            except:
                print(f"'{model_name}' model is not found in RedisAI")
                pred = None

        except:
            print(f"'{target_expr}' experiment is not found in MLflow")
            pred = None

        return pred

        

    def inference_caching(self, target_expr, input_tensor):
        ''' 
        input -> inference -> caching to redisai 
        '''
        print(f">> START INFERENCE '{target_expr}' EXPRIMENT'S MODEL ..")

        s = time.time()
        try:
            # Check latest model 
            model_name = self.get_latest_model(target_expr)

            try:
                # Check latest model from RedisAI
                model_meta = self.redisai_client.modelget(model_name, meta_only=True)

                # predict by redisai
                self.redisai_modelexecute(model_name, input_tensor)
            except:
                print(f"'{model_name}' model is not found in RedisAI")

        except:
            print(f"'{target_expr}' experiment is not found in MLflow")


    def get_cached_result(self, target_expr):
        ''' 
        retun cached result 
        '''
        try:
            # Check latest model 
            model_name = self.get_latest_model(target_expr)
            pred = self.redisai_client.tensorget(f'{model_name}:out1')

            # Get output from RedisAI
            if pred:
                return {
                    'prediction': pred[0], 
                    'model_info': {
                        'name':model_name, 
                        'version':latest_model.version, 
                        'source': latest_model.source,
                        'creation_timestamp': latest_model.creation_timestamp,
                        'last_updated_timestamp': latest_model.last_updated_timestamp
                    }
                }
            else:
                print(f"Output of '{model_name}' is not found in RedisAI")
                return {
                    'prediction': None, 
                    'model_info': {
                        'name':model_name, 
                        'version':latest_model.version, 
                        'source': latest_model.source,
                        'creation_timestamp': latest_model.creation_timestamp,
                        'last_updated_timestamp': latest_model.last_updated_timestamp
                    }
                }

        except:
            # print(f"{traceback.format_exc()}")
            print(f"'{target_agent}' experiment is not found in MLflow")
            return {
                'prediction': None, 
                'model_info': {
                    'name':None, 
                    'version':None, 
                    'source': None,
                    'creation_timestamp': None,
                    'last_updated_timestamp': None
                }
            }






