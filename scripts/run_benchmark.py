
import os
import shutil
import subprocess
import time

def run_command(cmd):
    print(f"\n[BENCHMARK] Running: {cmd}")
    start_time = time.time()
    try:
        subprocess.check_call(cmd, shell=True)
        print(f"[BENCHMARK] Command finished in {time.time() - start_time:.2f}s")
    except subprocess.CalledProcessError as e:
        print(f"[BENCHMARK] Command failed with error: {e}")
        exit(1)

def main():
    base_run_dir = os.path.join("runs", "phase1")
    
    # Needs to ensure clean start
    if os.path.exists(base_run_dir):
        print(f"Cleaning existing {base_run_dir}...")
        shutil.rmtree(base_run_dir)

    # --- Experiment A: Random Data (0% Expert) ---
    print("\n" + "="*50)
    print(" EXPERIMENT A: RANDOM DATA (0% EXPERT)")
    print("="*50)
    
    run_command("python scripts/generate_dataset.py --phase 1 --expert_prob 0.0")
    run_command("python scripts/train_pessimist.py --phase 1")
    run_command("python scripts/train_udrl.py --phase 1")
    
    # Move results
    random_dir = os.path.join("runs", "phase1_random")
    if os.path.exists(random_dir):
        print(f"Removing old {random_dir}...")
        shutil.rmtree(random_dir)
    
    print(f"Moving {base_run_dir} -> {random_dir}")
    shutil.move(base_run_dir, random_dir)
    
    # --- Experiment B: Mixed Data (50% Expert) ---
    print("\n" + "="*50)
    print(" EXPERIMENT B: MIXED DATA (50% EXPERT)")
    print("="*50)
    
    run_command("python scripts/generate_dataset.py --phase 1 --expert_prob 0.5")
    run_command("python scripts/train_pessimist.py --phase 1")
    run_command("python scripts/train_udrl.py --phase 1")
    
    # Move results
    mixed_dir = os.path.join("runs", "phase1_mixed")
    if os.path.exists(mixed_dir):
        print(f"Removing old {mixed_dir}...")
        shutil.rmtree(mixed_dir)
        
    print(f"Moving {base_run_dir} -> {mixed_dir}")
    shutil.move(base_run_dir, mixed_dir)

    print("\n" + "="*50)
    print(" BENCHMARK COMPLETE ")
    print("="*50)
    print(f"Random Experiment: {random_dir}")
    print(f"Mixed Experiment : {mixed_dir}")

if __name__ == "__main__":
    main()
