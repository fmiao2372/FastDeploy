# Intel HPU serving benchamrk

## 1. start server

```bash
./benchmark_paddle_hpu_server.sh
```
In the script, you can use FLAGS_selected_intel_hpus to select hpu card.

## 2. run benchmark cli

```bash
./benchmark_paddle_hpu_cli.sh
```

## 3. rarse logs
```python
python parse_benchmark_logs.py benchmark_fastdeploy_logs/[the targeted folder]
```
The performance data will be saved as a CSV file.