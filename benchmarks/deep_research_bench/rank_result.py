import json
from pathlib import Path

def rank_result(result_file:Path, output_file:Path, keyword:str='overall_score') -> None:
    temp_results = []
    with open(result_file, 'r') as f:
        temp_results = [json.loads(line) for line in f]
    temp_results.sort(key=lambda x: x[keyword], reverse=True)
    for i, result in enumerate(temp_results):
        result['rank'] = i + 1
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in temp_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    result_file = Path('/public/home/lnp/enterprise-deep-research/benchmarks/deep_research_bench/results/race/claude-3-7-sonnet-latest/raw_results.jsonl')
    output_file = Path('/public/home/lnp/enterprise-deep-research/benchmarks/deep_research_bench/results/race/claude-3-7-sonnet-latest/ranked_results.jsonl')
    rank_result(result_file, output_file)