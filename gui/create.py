from tkinter import *
from tkinter import ttk, filedialog
from tkinter.filedialog import askopenfile
import os
win = Tk()
win.geometry("700x350")

def open_file():
   file = filedialog.askopenfile(mode='r', filetypes=[('CSV files', '*.csv'),("Excel files","*.xlsx"),("All files","*.*")])
   if file:
      filepath = os.path.abspath(file.name)
      os.system("cp "+filepath+" /home/suganth/customer-segmentation/gui/Mall_Customers.csv")
      win.destroy()
      os.system("python3 progress.py")
label = Label(win, text="Click the Button to browse the Files", font=('Georgia 13'))
label.pack(pady=10)
ttk.Button(win, text="Browse", command=open_file).pack(pady=20)
win.mainloop()
