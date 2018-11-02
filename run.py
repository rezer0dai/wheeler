import os, time, threading

def watchdog():
    time.sleep(60 * 60)
    os.system("mv reacher_her_dtb/* backup/")
    time.sleep(1)
    os.system("sh run.sh")

os.system("mv backup/* reacher_her_dtb/")
time.sleep(1)
threading.Thread(target=watchdog).start()
os.system("python envs/reacher_her.py")
