import os
import webbrowser
import tkinter as tk
import numpy as np
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk,ImageOps, ImageFilter
from tkinter import messagebox

abs_file = os.path.abspath(__file__)
abs_dir = abs_file[:abs_file.rfind('\\')] if os.name == 'nt' else abs_file[:abs_file.rfind(r'/')]


#Get the current working path.
current_path = os.getcwd()
print("Current working directory:", current_path)
os.chdir(abs_dir)
from segment_lung_xrays_to_GUI import segment_lung_xrays, segment_model, device
from classify_lung_xrays_to_GUI import classify_lung_xrays, classify_model, dvc
#GUI background image path
#Reminder: You need to change the following file path to the true path in your computer.
background_image='background_image.png'


def show_readme():
    '''
    Define the "show_readme" function so that it can jump to our instruction section
    '''
    webbrowser.open("https://github.com/BIA4-course/2023-24-Group-07/blob/main/BIA4_Group7/BIA4_ICA1_Group07_Documentation.pdf")

def learn_more():
    '''
    Define the "learn_more" function, make it can be linked to the external websites of introduction of tuberculosis disease
    '''
    webbrowser.open('https://www.lung.org/lung-health-diseases/lung-disease-lookup/tuberculosis/treating-and-managing')
# Handle the function of "input" button click
# Process the input image with preprocessing,  lung segmentation and tuberculosis diagnosis
def segment_image(image_path, segment_model, device):
    '''
    Invoke the lung segmentation function that have been established before and deliver related parameters
    '''

    segmented_img = segment_lung_xrays([image_path], segment_model, device)


    segmented_img_pil = Image.fromarray((segmented_img[0] * 255).astype(np.uint8))
    return segmented_img_pil
    #Convert a NumPy array to a PIL image

def classify_image(image_path, classify_model, dvc):
    '''
    Define a "classify_image" function. Invoke the lung classification function that have been established before and deliver related parameters
    '''

    classified_scores = classify_lung_xrays([image_path], classify_model, dvc)

    # Diagnose the images and output the results
    # Define the classified_scores. If it has been diagnosed with tuberculosis, define as 1. If it has been diagnosed with normal, define as 0.

    if classified_scores == 1:
        classiication_result = 'Result: You most likely have tuberculosis!\n             Seek diagnosis and treatment now!'
    elif classified_scores == 0:
        classiication_result = 'Result: Congratulation! \n            You are unlikely to have tuberculosis.'
    else:
        classiication_result = 'There may be some errors in the operation of our software.'

    return classiication_result


def start_processing():

    #Pop up the file selection dialog box, provide the user to select the picture.
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpeg"), ("JPG files", "*.jpg")]
    )
    if file_path:
        # Display the user-input image

        img = Image.open(file_path)
        img.thumbnail((256, 256))
        # Convert the image into gray image and carry out the histogram equalization.
        img = img.convert('L')
        image = ImageOps.equalize(img)
        # Histogram equalization helps enhance the contrast of the image, making details in the image clearer.

        image = image.filter(ImageFilter.GaussianBlur(1))
        # Perform Gaussian blur processing on the image to smooth the image and reduce noise.
        # This can improve the visual quality of the image.
        image = image.resize((256,256))

        img= ImageTk.PhotoImage(image)
        #Convert a processed PIL image object image to a tkinter PhotoImage object.
        input_img_label.config(image=img)
        input_img_label.image = img

        explanation_text = "Processed Image:"        #Present text indicating that the image has been processed
        explanation_label.config(text=explanation_text)
        explanation_label.place(x=600, y=150)
        # Segment the image
        segmented_img = segment_image(file_path, segment_model, device)

        # Display the segmented image
        segmented_img.thumbnail((256, 256))
        segmented_img = ImageTk.PhotoImage(segmented_img)
        result_img_label.config(image=segmented_img)
        result_img_label.image = segmented_img

        result_label.config(text=classify_image(file_path, classify_model, dvc))
        #Call the classification function and output the diagnosis results
    else:
        # If no image is selected, clear or hide the explanatory text
        explanation_label.config(text="")
        explanation_label.place_forget()



# Create the main window
root = tk.Tk()
root.title("Image Processing GUI")

#Set background image
bg_image = Image.open(background_image)
bg_image = ImageTk.PhotoImage(bg_image)



# Create a Canvas widget to add the background image
canvas = tk.Canvas(root, width=bg_image.width(), height=bg_image.height())
canvas.pack()

# Place the background image on the Canvas
canvas.create_image(0, 0, anchor=tk.NW, image=bg_image)

explanation_label = ttk.Label(canvas, text="", style="TLabel")
explanation_label.place(x=600, y=150)
# Provides explanatory information about the status of image processing


def teamsinformation():
    messagebox.showinfo("About us", "Teams of Developers: \n"
                                    "Yingqi Li (yingqi.20@intl.zju.edu.cn), \n"    
                                    "If you have any further questions, please feel free to contact us!")
    # Show our team information

# Create a GUI window,and place a button in the window.
start_button = ttk.Button(canvas, text="Input", command=start_processing)
style = ttk.Style()

# Set button style, position and function
style.configure("TButton", padding=6, relief="flat", background="black", font=("Arial", 18))
start_button.place(x=540, y=500)

teams_info_button = ttk.Button(canvas, text="Contact us!", command=teamsinformation)
teams_info_button.place(x=890, y=630)

learn_button =ttk.Button(root,text="Learn More", command=learn_more, style="Custom.TButton")
learn_button.place(x=500, y=20)

link_button = ttk.Button(root, text="Instruction", command=show_readme)

link_button.place(x=200, y=630)

style = ttk.Style()

# Set lable style, position and function
style.configure("TLabel", font=("Arial", 22), foreground="black", background="white")

input_img_label = ttk.Label(canvas, text=" Loaded Image", style="TLabel")
input_img_label.place(x=500, y=180)
input_img_label.configure(background="white")


result_img_label = ttk.Label(canvas, text="Segemanted Image", style="TLabel")
result_img_label.place(x=800, y=180)
result_img_label.configure(background="white")

result_label = ttk.Label(canvas, text="Result: ", style="TLabel")
result_label.place(x=560, y=570)
result_label.configure(background="white")



# Run the Tkinter main loop
root.mainloop()
os.chdir(current_path)