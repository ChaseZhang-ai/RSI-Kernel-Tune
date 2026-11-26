# Paged MLA decode benchmark record

## GB300 initial direct and split-KV screen

Both candidates passed correctness and timing on all six screened official workloads. The split-KV variant reduced latency on the larger inputs, including 91.232 µs versus 829.584 µs for batch 64 with 75,145 pages. Both candidates shared the single-launch kernel on the eight-page workload. The full evaluator run is in progress. [Per-workload results](gb300-initial-screen.jsonl)

- Physical GPU 2, UUID `GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`, locked to 1500/3996 MHz SM/memory. PyTorch reported 152 SMs and CUDA capability 10.3.
- NVIDIA SOL-ExecBench 1.0.2, PyTorch 2.10.0+cu130, Triton 3.6.0. The runner used the official definition, trace metadata and tolerances, and measured median CUPTI first-to-last activity with 10 warmups, 50 timed iterations, cold L2, cloned inputs and preallocated destinations.
