import os

PROJECT_ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT_PATH, '_output')

if os.exists(MODEL_OUTPUT_PATH) == False:
	os.makedirs(MODEL_OUTPUT_PATH)