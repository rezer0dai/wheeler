import os, time, threading

def watchdog():
    time.sleep(60 * 60)
    os.system("sh reset")

threading.Thread(target=watchdog).start()
#os.system("python ddpg_reacher.py")
os.system("python reacher.py")
