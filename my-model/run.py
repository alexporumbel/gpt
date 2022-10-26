import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from syndicai import PythonPredictor

# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
sample_data = (
 "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
 "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
 "researchers was the fact that the unicorns spoke perfect English."
)


def run(opt):

    # Convert image url to JSON string
    sample_json = {"text": opt.text}

    # Run a model using PythonPredictor from syndicai.py
    model = PythonPredictor([])
    response = model.predict(sample_json)

    # Print a response in the terminal
    if opt.response:
        print(response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default=sample_data, type=str, help='URL to a sample input data')
    parser.add_argument('--response', default=True, type=bool, help='Print a response in the terminal')
    opt = parser.parse_args()
    run(opt)
