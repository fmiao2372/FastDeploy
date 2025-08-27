import matplotlib.pyplot as plt
import re
import sys
from datetime import datetime
import matplotlib.dates as mdates
import os
import csv

log_patterns = [
    re.compile(r'benchmarkdata_(.+?)_inputlength_(\d+)_outputlength_(\d+)_batchsize_(\d+)_numprompts_(\d+)_.*_profile\.log$'),
]

def draw_time_graph(log_dir, log_filename):
    # 用于存储提取的时间和BT值
    timestamps = []
    times = []
    bt_values = []
    block_list_shapes = []
    block_indices_shapes = []

    # 使用正则表达式提取 Model execution time 和 BT 信息
    pattern = re.compile(r'(\d+-\d+-\d+ \d+:\d+:\d+,\d+) .* Model execution time\(ms\): ([\d\.]+), BT=(\d+), block_list_shape=\[(\d+)\], block_indices_shape=\[(\d+)\]')

    # 读取日志文件
    with open(os.path.join(log_dir, log_filename), 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                timestamps.append(datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f"))
                times.append(float(match.group(2)))
                bt_values.append(int(match.group(3)))
                block_list_shapes.append(int(match.group(4)))
                block_indices_shapes.append(int(match.group(5)))

    # 绘制图表
    plt.figure(figsize=(15, 7))

    date_format = mdates.DateFormatter('%m-%d %H:%M:%S')
    # 绘制时间图
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, times, label='Model Execution Time (ms)')
    plt.ylabel('Execution Time (ms)')
    # plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.legend()

    # 绘制BT值图
    plt.subplot(2, 1, 2)
    plt.plot(timestamps, bt_values, label='BT', color='orange')
    plt.ylabel('BT Value')
    plt.xlabel(log_filename, fontsize=8)
    # plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.legend()

    plt.tight_layout()
    output_filename = log_filename[:-4] + '_analysis.png'
    plt.savefig(os.path.join(log_dir, output_filename), dpi=300)

    # 写入CSV文件
    csv_filename = log_filename[:-4] + '_analysis.csv'
    with open(os.path.join(log_dir, csv_filename), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Timestamp', 'ExecutionTime(ms)', 'BT', 'block_list_shape', 'block_indices_shape'])
        for i in range(len(times)):
            writer.writerow([
                timestamps[i],
                times[i],
                bt_values[i],
                block_list_shapes[i],
                block_indices_shapes[i]
            ])


def main():
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        log_dir = "."
    try:
        from natsort import natsorted
        natsort_available = True
    except ImportError:
        natsort_available = False
    all_files = set(os.listdir(log_dir))
    files = []
    for f in os.listdir(log_dir):
        for pat in log_patterns:
            if pat.match(f):
                files.append(f)
                break
    if natsort_available:
        files = natsorted(files)
    else:
        import re as _re
        def natural_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in _re.split('([0-9]+)', s)]
        files.sort(key=natural_key)
    rows = []

    for file in files:
        for idx, pat in enumerate(log_patterns):
            m = pat.match(file)
            if m:
                draw_time_graph(log_dir, file)

if __name__ == "__main__":
    print("Starting to draw logs...")
    main()
