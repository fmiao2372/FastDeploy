"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from fastdeploy import LLM, SamplingParams
import argparse

common_prefix = (
    "北京，中华人民共和国的首都，是一座融合了厚重历史与现代活力的超大城市。作为国家的政治中心、文化中心、国际交往中心和科技创新中心，北京承载着国家最高权力机关和众多国际机构。\n"
    "北京的历史可追溯至三千年前。它是元、明、清三朝古都，拥有众多举世闻名的文化遗产：世界上规模最大、保存最完整的古代宫殿建筑群——故宫，历经六百余年风雨；被誉为世界建筑奇迹的万里长城，"
    "其精华段蜿蜒于北京北部群山；庄严肃穆的天坛，是古代帝王祭天的圣地；贯穿城市南北、体现传统规划智慧的中轴线，串联起众多历史地标。\n"
    "步入现代，北京展现出蓬勃的活力。鸟巢（国家体育场）和水立方（国家游泳中心）是2008年奥运会的标志性遗产。中央商务区（CBD） 摩天大楼林立，彰显着经济实力。"
    "同时，传统的胡同和四合院依然散发着独特的生活气息，北京烤鸭等美食吸引着世界各地的游客。北京，这座古老而年轻的城市，正以其兼容并蓄的魅力，续写着辉煌篇章。\n"
    "阅读以上文字，回答下列问题"
)


prompts = [
    "北京作为中国的首都，主要承担着哪几个方面的中心职能？",
]

generating_prompts = [common_prefix + prompt for prompt in prompts]


sampling_params = SamplingParams(temperature=1, top_p=0.0, max_tokens=320)


parser = argparse.ArgumentParser(description="Offline Prefix Caching Demo")
parser.add_argument("--model", type=str, default="/data/disk2/ERNIE-4.5-21B-A3B-Paddle", help="Path to the model")
parser.add_argument("--engine_worker_queue_port", type=int, default=8182, help="Engine worker queue port")
parser.add_argument("--cache_queue_port", type=int, default=8183, help="Cache queue port")
parser.add_argument("--max_model_len", type=int, default=8192, help="Max model length")
parser.add_argument("--enable_prefix_caching", action="store_true", help="Enable prefix caching")
args = parser.parse_args()

prefix_cached_llm = LLM(
    model=args.model,
    enable_prefix_caching=args.enable_prefix_caching,
    engine_worker_queue_port=args.engine_worker_queue_port,
    cache_queue_port=args.cache_queue_port,
    max_model_len=args.max_model_len,
    num_gpu_blocks_override=6400
)

# ============================================================================================================================================
# prefix_outputs = prefix_cached_llm.generate(generating_prompts, sampling_params)
# for output in prefix_outputs:
#     prompt = output.prompt
#     print("prompt\n", prompt)
#     generated_text = output.outputs.text
#     print("generated_text", generated_text)
#     print("-" * 50)

# # generating_prompts.insert(0, common_prefix + "北京有哪些著名的历史遗迹？")
# # generating_prompts.append(common_prefix + "北京有哪些著名的历史遗迹？")
# generating_prompts.insert(0, "北京有哪些著名的历史遗迹？")
# generating_prompts.append("北京有哪些著名的历史遗迹？")
# prefix_outputs = prefix_cached_llm.generate(generating_prompts, sampling_params)

# # 输出结果
# for output in prefix_outputs:
#     prompt = output.prompt
#     print("prompt\n", prompt)
#     generated_text = output.outputs.text
#     print("generated_text", generated_text)
#     print("-" * 50)

# ============================================================================================================================================
LONG_PROMPT = (
    "You are a helpful assistant in recognizes the content of tables in markdown format. Here is a table as follows.\n# Table\n"
    + """
| ID  | Name          | Age | Occupation    | Country       | Email                  | Phone Number   | Address                       |
|-----|---------------|-----|---------------|---------------|------------------------|----------------|------------------------------|
| 1   | John Doe      | 29  | Engineer      | USA           | john.doe@example.com   | 555-1234       | 123 Elm St, Springfield, IL  |
| 2   | Jane Smith    | 34  | Doctor        | Canada        | jane.smith@example.com | 555-5678       | 456 Oak St, Toronto, ON      |
| 3   | Alice Johnson | 27  | Teacher       | UK            | alice.j@example.com    | 555-8765       | 789 Pine St, London, UK      |
| 26  | Xavier Green  | 34  | Scientist     | Canada        | xavier.g@example.com   | 555-9091       | 357 Oak St, Montreal, QC     |
| 27  | Yara Red      | 41  | Teacher       | UK            | yara.r@example.com     | 555-1214       | 975 Pine St, Leeds, UK       |
| 28  | Zack Blue     | 30  | Lawyer        | Australia     | zack.b@example.com     | 555-3436       | 135 Birch St, Adelaide, SA   |
| 29  | Amy White     | 33  | Musician      | New Zealand   | amy.w@example.com      | 555-5658       | 159 Maple St, Wellington, NZ |
| 30  | Ben Black     | 38  | Chef          | Ireland       | ben.b@example.com      | 555-7870       | 246 Fir St, Waterford, IE    |
"""
)

prefix_outputs = prefix_cached_llm.generate([LONG_PROMPT
        + "Question: what is the age of John Doe? Your answer: The age of John Doe is "], sampling_params)
for output in prefix_outputs:
    prompt = output.prompt
    print("prompt\n", prompt)
    generated_text = output.outputs.text
    print("generated_text", generated_text)
    print("-" * 50)

prefix_outputs = prefix_cached_llm.generate([LONG_PROMPT
        + "Question: what is the address of Jane Smith? Your answer: The address of Jane Smith is "], sampling_params)
for output in prefix_outputs:
    prompt = output.prompt
    print("prompt\n", prompt)
    generated_text = output.outputs.text
    print("generated_text", generated_text)
    print("-" * 50)

prefix_outputs = prefix_cached_llm.generate(["What is Artificial Intelligence?", LONG_PROMPT
        + "Question: what is the email of Alice Johnson? Your answer: The email of Alice Johnson is ", "What is Artificial Intelligence?"], sampling_params)

# 输出结果
for output in prefix_outputs:
    prompt = output.prompt
    print("prompt\n", prompt)
    generated_text = output.outputs.text
    print("generated_text", generated_text)
    print("-" * 50)
