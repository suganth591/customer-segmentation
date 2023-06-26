import tkinter as tk
from tkinter import *
from tkinter import ttk, filedialog
import threading
from tkinter.filedialog import askopenfile
import os
import time
import sys
i=1
def f5():
    quit()
def f4():
    global ttt,i
    try:
        path=str(os.getcwd())+"/completed.txt"
        f=open(path,'r')
        f.close()
    except:
        print("NO "+str(i))
        if(i!=0):
            os.system("python3 initial.py")
            i=0
        print("NO "+str(i))
        sys.exit(1)
    finally:
        time.sleep(1)
        if(i==1):
            f4()
        else:
            sys.exit(0)

def f3():
    path=str(os.getcwd())+"/completed.txt"
    i=9
    global rootpro
    while(True):
        print("Testing.....")
        f=open(path,'r')
        if(f.readline()=="OK"):
            ttt.start()
            print("OK")
            f.seek(0)
            os.system("rm completed.txt")
            rootpro.destroy()
            i=5
            break
        time.sleep(1)
        print("Where is file in "+path)
        continue
def getit():
    f = open("file.txt", "w+")
    f.write(e.get())
    f.close()
    first.destroy()
#def f1():
def f2():
    os.system("python3 ../project.py")
first = tk.Tk()
first.geometry("500x300")
r=tk.Label(first, text="Save result as:")
r.pack()
e = tk.Entry(first)
button1 = tk.Button(text='Save Report', command=getit)
e.pack()
button1.pack()
first.mainloop()
f=open("completed.txt","w")
f.write("NO")
ttt=threading.Thread(target=f4)
threading.Thread(target=f2).start()
threading.Thread(target=f3).start()
rootpro = Tk()
label = Label(rootpro, text ='Progress Bar', font = "50")
label.pack(pady=5)
progbar = ttk.Progressbar(rootpro, orient=HORIZONTAL, length=220, mode="indeterminate")
progbar.pack(pady=20)
rootpro.geometry("300x150")
rootpro.title("Creating report")
progbar.start()
path=str(os.getcwd())+"/completed.txt"
progbar.mainloop()