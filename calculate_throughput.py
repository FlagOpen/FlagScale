import re
import argparse

def calculate_throughput(log_file, start_iter, end_iter):
    """
    Calculate samples per GPU per second between two iterations from a log file.
    """
    world_size = None
    iteration_data = {}

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Find world size
                if world_size is None:
                    match = re.search(r'using world size: (\d+),', line)
                    if not match:
                        match = re.search(r'world_size\s+\.+\s+(\d+)', line)
                    if match:
                        world_size = int(match.group(1))

                # Find iteration data
                iter_match = re.search(r'iteration\s+(\d+)/\s*\d+\s*\|\s*consumed samples:\s*([\d\.]+)\s*\|\s*elapsed time per iteration \(ms\):\s*([\d\.]+)', line)
                if iter_match:
                    iteration = int(iter_match.group(1))
                    consumed_samples = int(float(iter_match.group(2)))
                    elapsed_time_ms = float(iter_match.group(3))
                    iteration_data[iteration] = {
                        'consumed_samples': consumed_samples,
                        'elapsed_time_ms': elapsed_time_ms
                    }
    except FileNotFoundError:
        print(f"Error: Log file not found: {log_file}")
        return
    except Exception as e:
        print(f"Error reading or parsing log file: {e}")
        return

    if world_size is None:
        print("Error: Could not find world size in the log file.")
        return

    if not iteration_data:
        print("Error: Could not find any iteration data in the log file.")
        return
        
    if start_iter > end_iter:
        print("Error: Start iteration must be less than or equal to end iteration.")
        return

    if start_iter not in iteration_data or end_iter not in iteration_data:
        print(f"Error: Iteration range [{start_iter}, {end_iter}] is not complete in the log file.")
        available_iters = sorted(iteration_data.keys())
        print(f"Available iterations: from {available_iters[0]} to {available_iters[-1]}")
        return

    # Calculate consumed samples difference
    consumed_samples_at_end = iteration_data[end_iter]['consumed_samples']
    
    if start_iter == 1:
        consumed_samples_before_start = 0
    else:
        if (start_iter - 1) not in iteration_data:
            print(f"Error: Could not find data for Iteration {start_iter - 1} to calculate consumed samples difference.")
            return
        consumed_samples_before_start = iteration_data[start_iter - 1]['consumed_samples']
    
    consumed_samples_diff = consumed_samples_at_end - consumed_samples_before_start

    # Calculate sum of elapsed time
    total_elapsed_time_ms = 0
    for i in range(start_iter, end_iter + 1):
        if i not in iteration_data:
            print(f"Error: Missing data for iteration {i}.")
            return
        total_elapsed_time_ms += iteration_data[i]['elapsed_time_ms']
        
    total_elapsed_time_s = total_elapsed_time_ms / 1000

    if total_elapsed_time_s == 0:
        print("Error: Total elapsed time is zero, cannot calculate throughput.")
        return

    # Calculate Samples per GPU per second
    samples_per_gpu_per_second = consumed_samples_diff / total_elapsed_time_s / world_size

    print(f"Log file: {log_file}")
    print(f"World Size: {world_size}")
    print(f"Iteration range: {start_iter} to {end_iter}")
    print("-" * 30)
    print(f"Total consumed samples in range: {consumed_samples_diff}")
    print(f"Total elapsed time in range: {total_elapsed_time_s:.2f} seconds")
    print(f"Samples per GPU per second (Samples/GPU/sec): {samples_per_gpu_per_second:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Samples per GPU per second from a log file.")
    parser.add_argument('log_file', type=str, help='Path to the log file.')
    parser.add_argument('start_iter', type=int, help='The starting iteration number.')
    parser.add_argument('end_iter', type=int, help='The ending iteration number.')

    args = parser.parse_args()

    calculate_throughput(args.log_file, args.start_iter, args.end_iter) 
