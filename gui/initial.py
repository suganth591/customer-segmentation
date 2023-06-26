from tkinter import *
from tkinter import ttk, filedialog
from tkinter.filedialog import askopenfile
import time
import os
def createit():
    wini.destroy()
    os.system("python3 create.py")
def view():
    wini.destroy()
    os.system("python3 view.py")
wini = Tk()
wini.geometry("700x350")
create=Button(text="Create New Report",command=createit)
create1=Button(text="View Previous reports",command=view)
create.pack(pady=20)
create1.pack(pady=20)
wini.mainloop()
