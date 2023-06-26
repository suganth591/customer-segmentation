from tkinter import *
import os
root = Tk()
root.geometry( "400x300" )
def show():
    """
    os.system("python3 report.py "+clicked.get())
    """
    os.system("chromium output/"+clicked.get()+"/index.html")
    root.destroy()
    os.system("python3 initial.py")
    label.config( text = "You are viewing "+clicked.get() )
options = os.popen("ls output/").read().split('\n')
clicked = StringVar()
clicked.set( options[0] )
drop = OptionMenu( root , clicked , *options )
drop.pack(pady=20)
button = Button( root , text = "Open Report" , command = show ).pack()
label = Label( root , text = " " )
label.pack(pady=20)
root.mainloop()
