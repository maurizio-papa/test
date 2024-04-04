import subprocess
import time

# List of participant numbers
participants = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]

# Loop through each participant and start the download process in the background
for participant in participants:
    # Command to execute the downloader script in the background
    command = f"python3 epic_downloader.py --videos --participants {participant}"
    
    # Execute the command in the background
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Print feedback or handle the process as needed
    print(f"Started download process for participant {participant}")

# Sleep to allow some time for the processes to start
    time.sleep(1)
