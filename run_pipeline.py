"""
AUTOMATED IIOT ANOMALY DETECTION PIPELINE MANAGEMENT SUITE
==========================================================
Description:
    Manages end-to-end extraction, asset layout validation, execution,
    and reporting cycles for the Anomal-E verification framework.
"""

import os
import sys
import time
import zipfile
import urllib.request
import subprocess

def download_and_extract_dataset(target_dir="data", csv_filename="wustl_iiot_2021.csv"):
    csv_path = os.path.join(target_dir, csv_filename)
    if os.path.exists(csv_path):
        print(f"--> [Asset Found] Verified presence of {csv_filename} in target directory.")
        return

    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "wustl_iiot_2021.zip")
    source_url = "https://www.cse.wustl.edu/~jain/iiot2/wustl_iiot_2021.zip"
    
    print(f"--> [Asset Missing] Target {csv_filename} not found.")
    print(f"--> [Download Initiation] Fetching archive from primary remote repository...")
    
    try:
        urllib.request.urlretrieve(source_url, zip_path)
        print(f"--> [Download Verification] Download completed successfully.")
        
        print(f"--> [Extraction Phase] Unzipping archive layers...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
        print(f"--> [Asset Ready] Dataset configuration fully initialized.")
    except Exception as error:
        print(f"❌ [Network Exception] Failed to download or unzip dataset: {error}")
        sys.exit(1)

def run_script(script_name):
    print(f"\n🚀 [Active Stage Change] Launching subsystem module: {script_name}")
    start_time = time.time()
    
    process = subprocess.Popen(
        [sys.executable, script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    if process.stdout:
        for line in process.stdout:
            print(f"  [{script_name}] {line.strip()}")
            
    process.wait()
    elapsed = time.time() - start_time
    
    if process.returncode == 0:
        print(f"✅ [Stage Success] {script_name} finalized operations inside {elapsed:.2f} seconds.")
    else:
        print(f"❌ [Stage Exception] Critical failure within {script_name}. Process aborted.")
        sys.exit(process.returncode)

def main():
    print("=====================================================================")
    print("🤖 STARTING AUTOMATED IIOT ANOMALY DETECTION BENCHMARK SUITE")
    print("=====================================================================")
    global_start = time.time()

    download_and_extract_dataset()

    run_script("main.py")
    run_script("plot.py")

    total_elapsed = time.time() - global_start
    print("\n=====================================================================")
    print(f"🏁 PIPELINE RUN COMPLETE! Total Execution Time: {total_elapsed/60:.2f} minutes.")
    print("=====================================================================")

if __name__ == "__main__":
    main()