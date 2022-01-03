from typing import Any, List
import json
import math
import random
import argparse
import logging
from pandas import json_normalize
import torch
from transformers import GPTJForCausalLM, AutoTokenizer, GPTNeoForCausalLM
from utils import read_file, chunks
from frequency_data_utils import Num1MultiplyFactory, Num2MultiplyFactory, DayNum1ConvertFactory
from datapoint_utils import DataPointGenerator, DigitLimitFilter, MultiplyTemplate, Day1Template
from testcase_utils import TestCase, TestCaseResult, TestCaseTemplate, testcase_generator
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
    for testcase_chunk in chunks(testcases, batch_size):
        testcases_bodies = [x.body for x in testcase_chunk]
        input_ids = tokenizer.batch_encode_plus(testcases_bodies, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            generated_ids = model.generate(input_ids=input_ids['input_ids'], attention_mask=input_ids['attention_mask'],
                                           max_length=5 + len(input_ids['input_ids'][0]), do_sample=False, min_length=2)
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_answers = [x[len(testcases_bodies[i]) + 1:].split()[0] for i, x in enumerate(generated_texts)]
        for (tc, ans) in zip(testcase_chunk, generated_answers):
            results.extend([TestCaseResult(tc, ans, ans.strip() == tc.data_point.answer.strip())])
    return results


def main(args):
    # -------------- Setup -------------
    if args.local_rank == -1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda', args.local_rank)
    logger.info(f"Device is detected as {device}!")
    model, tokenizer = setup_model(device, args.model)
    random.seed(args.seed)  # TODO: set all
    torch.manual_seed(args.seed)
    logger.info("Model is loaded!")
    # ----------------------------------
    # ----- Read frequency data  -------
    complete_frequency_factory = Num1MultiplyFactory(list(range(1, 21)))
    if args.factory_type == "Num1":
        complete_frequency_factory = Num1MultiplyFactory(list(range(1, 21)))
    elif args.factory_type == "Num2":
        complete_frequency_factory = Num2MultiplyFactory()
    elif args.factory_type == "Day1":
        complete_frequency_factory = DayNum1ConvertFactory()
    input_path = args.input_path
    logger.debug(f"The input path is '{input_path}'")
    my_file = read_file(input_path)
    frequency_data = []
    for line in my_file.split("\n")[:args.top]:
        key, frequency = eval(line)
        frequency_data.extend(complete_frequency_factory.build(key, frequency))
    logger.info(f"Reading is done! Total Complete Frequency Data: {len(frequency_data)}")
    logger.debug("A sample of complete frequency data:")
    logger.debug("\n".join(str(x) for x in frequency_data[:40]))
    # ----------------------------------
    # -----  Generate DataPoints  ------
    datapoint_generator = DataPointGenerator()
    datapoint_generator.add_filter(DigitLimitFilter(2))
    if args.dp_template == "Mult":
        datapoint_generator.add_template(MultiplyTemplate())
    elif args.dp_template == "Day1":
        datapoint_generator.add_template(Day1Template())
    else:
        datapoint_generator.add_template(MultiplyTemplate())
    datapoints = datapoint_generator.generate(frequency_data)
    logger.info(f"Generated {len(datapoints)} data points!")
    logger.debug("A sample of data points:")
    logger.debug("\n".join(str(x) for x in datapoints[:40]))
    # ----------------------------------
    # -----  Generate TestCases  -------
    testcase_template = TestCaseTemplate("Q:", "A:", "Calculate the math question below:")
    testcases = testcase_generator(datapoints, testcase_template, args.shots)
    logger.info(f"Generated {len(testcases)} Test Cases!")
    logger.debug("A sample of test cases:")
    logger.debug("\n------\n".join(x.body for x in testcases[:3]))
    # ----------------------------------
    # --------  GPTJ Analysis  ---------
    results = GPTJ_Analysis(model, tokenizer, device, args.bs, testcases)
    logger.info(f"Finished: {len(results)} test case is analyzed!")
    # ----------------------------------
    # --------  Write outputs  ---------
    json_results = MyEncoder().encode(results)
    flatten_json_result = json_normalize(json.loads(json_results))
    flatten_json_result.to_csv(args.output_path)
    logger.info(f"The final result is written in '{args.output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True, help="The s3 or local path to the aggregated result")
    parser.add_argument('--output-path', type=str, required=True, help="The local path to write the output")
    parser.add_argument('--factory-type', type=str, default="Num1")
    parser.add_argument('--dp-template', type=str, default="Mult")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--bs', type=int, default=20, help="Batch size")
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
