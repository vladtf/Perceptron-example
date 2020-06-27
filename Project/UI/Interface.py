from tkinter import *
from Project.AI.Sigmoid import *
from tkinter import messagebox

root = Tk()
root.title('AI learning interface')
root.geometry("600x400")

myLabel = Label(root, text="Hello")
myLabel.pack()

myTextBox = Entry(root, width=30)
myTextBox.pack()

result_label = Label(root)
result_label.pack()


def myButtonClick():
    data = myTextBox.get()
    try:
        tokens = list(map(int, data.split(" ")))
        output = evaluate(tokens[0], tokens[1], tokens[2])
        result_label.config(text=output)
        if output == 1:
            root.config(bg='green')
        else:
            root.config(bg='red')

    except:
        messagebox.showerror("Error", "invalid input")


myTextBox.bind("<Return>", (lambda event: myButtonClick()))

myButton = Button(root, text="Submit", command=myButtonClick)
myButton.pack()

root.mainloop()
