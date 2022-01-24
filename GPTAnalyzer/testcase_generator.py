from typing import Any, List
from collections import defaultdict
import json
import math
import random
import argparse
import logging
from pandas import json_normalize, DataFrame
import torch
import os.path
from os import path

from transformers import GPTJForCausalLM, AutoTokenizer, GPTNeoForCausalLM


import utils
import frequency_data_utils
from datapoint_utils import DataPointGenerator, DigitLimitFilter
from testcase_utils import TestCase, TestCaseResult, TestCaseTemplate, testcase_generator, special_testcase_generator

logger = logging.getLogger(__name__)


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__


def setup_model(device, model_name: str) -> (Any, Any):
    if model_name == 'GPTNEO-small':
        logger.debug("in first model")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to(device)
    elif model_name == 'GPTNEO-large':
        logger.debug("in second model")
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
        model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B').to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
        model = GPTJForCausalLM.from_pretrained('EleutherAI/gpt-j-6B').to(device)
    tokenizer.padding_side = "left"  # TODO: check this please
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return (model, tokenizer)


def GPTJ_Analysis(model, tokenizer, device, batch_size: int, testcases: List[TestCase]) -> List[TestCaseResult]:
    results = []
    for testcase_chunk in utils.chunks(testcases, batch_size):
        testcases_bodies = [x.body for x in testcase_chunk]
        input_ids = tokenizer.batch_encode_plus(testcases_bodies, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            generated_ids = model.generate(input_ids=input_ids['input_ids'], attention_mask=input_ids['attention_mask'],
                                           max_length=5 + len(input_ids['input_ids'][0]), do_sample=False, min_length=2) #TODO think about the max len
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_answers = [x[len(testcases_bodies[i]) + 1:].split()[0] for i, x in enumerate(generated_texts)]
        for (tc, ans) in zip(testcase_chunk, generated_answers):
            results.extend([TestCaseResult(tc, ans, ans.strip() == tc.data_point.answer.strip())])
    return results


def aggregate_by_key(results: List[TestCaseResult]) -> List:
    total_map = defaultdict(int)
    correct_map = defaultdict(int)
    for result in results:
        key_freq = (result.testcase.data_point.frequency_data.key, result.testcase.data_point.frequency_data.frequency)
        total_map[key_freq] += 1
        if result.is_correct:
            correct_map[key_freq] += 1
    ag_result = []
    for key_freq in total_map.keys():
        ag_result.append({"Key": key_freq[0], "Frequency": key_freq[1], "#Correct": correct_map[key_freq],
                          "#Total": total_map[key_freq], "Accuracy": correct_map[key_freq] / total_map[key_freq]})
    return ag_result


def main(args):
    # -------------- Setup model-------------
    if args.local_rank == -1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda', args.local_rank)
    logger.info(f"Device is detected as {device}!")
    model, tokenizer = setup_model(device, args.model)
    logger.info("Model is loaded!")

    #check output file
    if path.exists(args.output_path) and args.overwrite:
        logger.info("overwrites the output file")
        open(args.output_path, 'w').close()
    elif path.exists(args.output_path) and args.append_output:
        logger.info(f"appending the results is {args.output_path}")
    elif not path.exists(args.output_path):
        open(args.output_path, 'w').close()
        logger.info(f"appending the results is {args.output_path}")
    else:
        assert False, 'output file exists, put --overwrite or --append_output'

    factory_types = [args.factory_type]
    if args.factory_type == 'Num1MoreLess':
        factory_types = ['Num1More', 'Num1Less']
        # multi_factory_type_analysis(args, ['Num1More', 'Num1Less'], model, tokenizer, device)
        # return

    # ----------------------------------
    # ----- Read frequency data  -------
    frequency_data_map = {}
    for factory_type in factory_types:
        frequency_data = []
        complete_frequency_factory = frequency_data_utils.create_complete_frequency_factory(factory_type)
        input_path = args.input_path
        logger.debug(f"The input path is '{input_path}'")
        my_file = utils.read_frequency_file(input_path)
        for line in my_file.split("\n")[:args.top]:
            key, frequency = eval(line)
            frequency_data.extend(complete_frequency_factory.build(key, frequency))
        frequency_data_map[factory_type] = frequency_data
    logger.info(f"Reading is done! Total Complete Frequency Data: {sum(len(x) for x in frequency_data_map.values())}")
    # logger.debug("A sample of complete frequency data:")
    # logger.debug("\n".join(str(x) for x in frequency_data[:40]))

    for i in range(args.number_of_seeds):
        utils.set_seed(i)
        # ----------------------------------
        # -----  Generate DataPoints  ------
        list_of_data_points = []
        for factory_type, frequency_data in frequency_data_map.items():
            data_points = DataPointGenerator() \
                .add_filter(DigitLimitFilter(2)) \
                .add_template(factory_type) \
                .generate(frequency_data)
            logger.info(f"Generated {len(data_points)} data points!")
            logger.debug("A sample of data points:")
            logger.debug("\n".join(str(x) for x in data_points[:40]))
            list_of_data_points.append(data_points)
        # # ----------------------------------
        # # -----  Generate TestCases  -------
        testcase_template = TestCaseTemplate("Q:", "A:", None)
        testcases = special_testcase_generator(list_of_data_points, testcase_template, args.shots)
        logger.debug("A sample of test cases:")
        logger.debug("\n------\n".join(x.body for x in testcases[:3]))
        # ----------------------------------
        # --------  GPTJ Analysis  ---------
        results = GPTJ_Analysis(model, tokenizer, device, args.bs, testcases)
        logger.info(f"Finished: {len(results)} test case is analyzed!")
        # ----------------------------------
        # --------  Write outputs  ---------
        # All Result
        json_results = MyEncoder().encode(results)
        flatten_json_result = json_normalize(json.loads(json_results))
        with open(args.output_path, 'a') as f:
            flatten_json_result.to_csv(f, mode='a', header=f.tell() == 0)
        # Aggregated Result
        ag_file_path = args.output_path.split(".")
        ag_file_path.insert(-1, "agg")
        ag_file_path = ".".join(ag_file_path)
        with open(ag_file_path, 'a') as f:
            DataFrame(aggregate_by_key(results)).to_csv(f, mode='a', header=f.tell() == 0)
        logger.info(f"The final result is written in '{args.output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True,
                        help="The s3 or local path to the aggregated result")
    parser.add_argument('--output-path', type=str, required=True, help="The local path to write the output")
    parser.add_argument('--overwrite', action='store_true', help='if specified, overwrites the output file')
    parser.add_argument('--append-output', action='store_true', help='if specified, appends to the existing output file')
    parser.add_argument('--factory-type', type=str, default="Num1")
    parser.add_argument('--dp-template', type=str, default="Mult") # Todo: currently not using this check if needed
    parser.add_argument('--number-of-seeds', type=int, default=5)
    parser.add_argument('--bs', type=int, default=50, help="Batch size")
    parser.add_argument('--shots', type=int, default=2)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--top', type=int, default=200, help="Filter the top frequent data")
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--model', type=str, default='GPT-J', help="model-name")

    args = parser.parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    main(args)
