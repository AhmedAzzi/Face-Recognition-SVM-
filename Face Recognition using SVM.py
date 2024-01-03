from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Progressbar
import tkinter.simpledialog as simpledialog
import cv2
import os
import face_recognition
from sklearn import svm
from PIL import Image, ImageTk


def train_svm():
    """Train a Support Vector Machine (SVM) classifier.
     This function creates a Tkinter window to show the progress of the training process.
    It also loads all the images from the 'train/' directory, encodes them and uses them to train the SVM classifier.
    """
    # Create a Tkinter window to show the progress of the training process
    progress_window = Tk()
    progress_window.title("Training SVM")
    progress_window.geometry("300x100+{}+{}".format(int(progress_window.winfo_screenwidth() /
                                                        2 - 100), int(progress_window.winfo_screenheight()/2 - 25)))
    progress_window.config(bg="white")
    progress_label = Label(
        progress_window, text="Training SVM...", bg="white", fg="black")
    progress_label.pack(padx=10, pady=5)
    progress_bar = Progressbar(
        progress_window, orient="horizontal", length=200, mode="determinate")
    progress_bar.pack(padx=10, pady=5)
    progress_bar["maximum"] = 100
    # Initialize the list of encodings and names
    encodings = []
    names = []
    # Get the list of people from the 'train/' directory
    train_dir = os.listdir('train/')
    progress_window.update_idletasks()
    # Iterate over each person in the 'train/' directory
    for i, person in enumerate(train_dir):
        # Get the list of pictures for each person
        pictures = os.listdir("train/" + person)
        # Iterate over each picture
        for person_img in pictures:
            # Load the image
            face = face_recognition.load_image_file(
                "train/" + person + "/" + person_img)
            # Find the face locations in the image
            face_bounding_boxes = face_recognition.face_locations(face)
            # If there is only one face in the image
            if len(face_bounding_boxes) == 1:
                # Encode the face
                face_encoding = face_recognition.face_encodings(face)[0]
                # Append the encoding and the name to the lists
                encodings.append(face_encoding)
                names.append(person)
            else:
                # Print a message if there are no faces or more than one face in the image
                print(person + "/" + person_img +
                    " was skipped and can't be used for training")
        # Update the progress bar and label
        progress_label.config(
            text="Processing {} ({}/{})".format(person, i+1, len(train_dir)))
        progress_bar["value"] = ((i+1)/len(train_dir))*100
        progress_window.update_idletasks()
    # Create the SVM classifier
    classifier = svm.SVC(gamma='scale')
    # Train the classifier
    classifier.fit(encodings, names) # type: ignore
    # Destroy the progress window
    progress_window.destroy()
    # Return the trained classifier
    return classifier

# Function to test a local image
def test_local_image(clf):
    # Create a Tk root window
    root = Tk()
    # Hide the main window
    root.withdraw()
    # Ask user to select an image file
    file_path = filedialog.askopenfilename()

    # If a file is selected
    if file_path:
        # Load the selected image
        test_image = face_recognition.load_image_file(file_path)
        # Find all the faces in the image
        face_locations = face_recognition.face_locations(test_image)
        # Count the number of faces
        num_faces = len(face_locations)
        print("Number of faces detected: ", num_faces)
        list_names = []
        print("Found:")
        # Loop through each face
        for i in range(num_faces):
            # Get the encoding of the face
            test_image_encoding = face_recognition.face_encodings(test_image)[
                i]
            # Use the classifier to predict the name of the face
            name = clf.predict([test_image_encoding])
            print(name)
            # Append the name to the list
            list_names.append(*name)
        # Display the test image with rectangles around the detected faces and their names
        display_name(list_names, file_path)
    else:
        print("No file selected.")

# Function to capture images for face recognition training
def train_img_capture():
    # Initialize image counter and file name
    img_counter = 0
    file_name = ''
    # Create a dialog box to get user input
    file_name = simpledialog.askstring(
        title="Face Recognition", prompt="What's your Name?:")
    # Create a window object
    window = Tk()
    window.withdraw()
    # Create a VideoCapture object
    cam = cv2.VideoCapture(0)
    # Create a named window
    cv2.namedWindow("Face Training")
    # If user input is not empty, create a directory with the user's name
    if file_name is not None and file_name.strip() != "":
        os.mkdir("train/"+file_name)
        print("OK button clicked. User's input:", file_name)
    else:
        # Release the camera and destroy all windows if user input is empty
        cam.release()
        cv2.destroyAllWindows()
        print("Cancel button clicked or dialog closed.")
        return
     # Capture images until user presses escape key
    while True:
        # Read frame from camera
        ret, frame = cam.read()
        # Check if frame was successfully grabbed
        if not ret:
            print("Failed to grab frame")
            break
         # Show the frame
        cv2.imshow("Face Training", frame)
        # Get key pressed
        k = cv2.waitKey(1)
        # If escape key is pressed, close the program
        if k % 256 == 27 or cv2.getWindowProperty("Face Training", cv2.WND_PROP_VISIBLE) == 0:
            print("Escape hit, closing...")
            break
         # If spacebar is pressed, save the image
        elif k % 256 == 32:
            img_name = file_name+"_{}.jpg".format(img_counter)
            cv2.imwrite(os.path.join("train/"+file_name, img_name), frame)
            print("{} written!".format(img_name))
            img_counter += 1
    # Release the camera and destroy all windows
    cam.release()
    cv2.destroyAllWindows()
    window.destroy()

def test_svm(clf):
    """
    This function tests the SVM classifier on a given image.
    Args:
        clf: The SVM classifier to be tested.
    """
    # Load the test image
    test_image = face_recognition.load_image_file('test/test.jpg')
    # Get the locations of the faces in the image
    face_locations = face_recognition.face_locations(test_image)
    # Get the number of faces detected
    num = len(face_locations)
    print("Number of faces detected: ", num)
    # Initialize an empty list to store the names of the faces detected
    list_names = []
    print("Found:")
    # Loop through each face detected
    for i in range(num):
        # Get the encoding of the face
        test_image_enc = face_recognition.face_encodings(test_image)[i]
        # Use the SVM classifier to predict the name of the face
        name = clf.predict([test_image_enc])
        print(name)
        # Append the name to the list
        list_names.append(*name)
    # Display the names of the faces detected
    display_name(list_names, "test/test.jpg")

# Function to capture an image for testing
def test_img_capture(clf):
    # Destroy the window
    window.destroy()
    img_counter = 0
    if (True):
        # Initialize the camera
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("Face Recognition")
        while True:
            # Read the frame from the camera
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            # Show the frame
            cv2.imshow("Face Recognition", frame)
            k = cv2.waitKey(1)
            # If escape is pressed, close the window
            if k % 256 == 27 or cv2.getWindowProperty("Face Recognition", cv2.WND_PROP_VISIBLE) == 0:
                print("Escape hit, closing...")
                break
            # If space is pressed, save the image and close the window
            elif k % 256 == 32:
                img_name = "test.jpg"
                cv2.imwrite(os.path.join('test', img_name), frame)
                print("{} written!".format(img_name))
                print("Closing now")
                img_counter += 1
                break
    # Release the camera
    cam.release()
    # Destroy all windows
    cv2.destroyAllWindows()
    # Call the SVM function
    test_svm(clf)
    
# Function to display names on faces in an image
def display_name(list_name, path):
    # Read the image from the given path
    img = cv2.imread(path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Get the face detector
    face_detector = dlib.get_frontal_face_detector() # type: ignore
    # Detect faces in the image
    faces = face_detector(gray)
    # Iterate over each face and draw a rectangle around it
    for face, name in zip(faces, list_name):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Put the name of the person on the image
        cv2.putText(img, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
     # Display the output image
    cv2.imshow('Output', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model = train_svm()  # Train the SVM model

    window = Tk()  # Create a window
    window.title("Face Recognition")
    window_width = 750
    window_height = 545

    # Set the window's position to the center of the screen
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    window.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    # Create a background image
    bg_img = Image.open(r'BG.jpg')
    bg_img = bg_img.resize((window_width, window_height), Image.BICUBIC)
    bg_img = ImageTk.PhotoImage(bg_img)
    bg_label = Label(window, image=bg_img)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    
    # Create a label with welcome message
    welcome = Label(window, text='WELCOME TO FACE RECOGNITION.\nSELECT AN OPTION BELOW:', font=(
        "Helvetica", 14, "bold"), bg='#181654', fg='white')
    welcome.pack(pady=20)

    # Create a button for training
    train = Button(window, text="Train", font=("Helvetica", 14, "bold"), bg="#A86C10", fg="#fff", activebackground="#226a00",
                   activeforeground="#000", highlightthickness=0, bd=0, width=15, height=2, cursor="hand2", command=train_img_capture)
    train.pack(pady=10)

    # Create a button for testing with updated model
    test_update = Button(window, text="Test (with updated model)", font=("Helvetica", 14, "bold"), bg="#A86C10", fg="#fff", activebackground="#226a00",
                         activeforeground="#000", highlightthickness=0, bd=0, width=25, height=2, cursor="hand2", command=lambda: test_img_capture(
        train_svm()))
    test_update.pack(pady=20)

    # Create a button for testing with existing model
    test_exist = Button(window, text="Test (with existing model)", font=("Helvetica", 16, "bold")
, bg="#A86C10", fg="#fff", activebackground="#226a00",
                        activeforeground="#000", highlightthickness=0, bd=0, width=25, height=2, cursor="hand2", command=lambda: test_img_capture(
        model))
    test_exist.pack(pady=20)

    # Create a button for testing with existing model
    test_local = Button(window, text="Test (with Local Image)", font=("Helvetica", 14, "bold"), bg="#A86C10", fg="#fff", activebackground="#226a00",
                        activeforeground="#000", highlightthickness=0, bd=0, width=25, height=2, cursor="hand2", command=lambda: test_local_image(
        model))
    test_local.pack(pady=20)

    # Create a label with instructions
    guide = Label(window, text="Instructions", font=(
        "Helvetica", 16, "bold"), bg='#181654', fg='white')
    guide.pack()
    guide = Label(window, text="1).In Train Mode, enter your name and press SPACEBAR to capture images. Hit ESC when done.\n2).In Test Mode, press SPACEBAR to capture image and display detected face", font=(
        "Helvetica", 12), bg='#181654', fg='white')
    guide.pack()

    window.mainloop()