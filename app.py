import streamlit as st

import cv2

import face_recognition

# Define known encodings

encodings = {

    "John Doe": face_recognition.face_encodings(face_recognition.load_image_file("john_doe.jpg"))[0],

    "Jane Smith": face_recognition.face_encodings(face_recognition.load_image_file("jane_smith.jpg"))[0],

    "Bob Johnson": face_recognition.face_encodings(face_recognition.load_image_file("bob_johnson.jpg"))[0]

}

# Define helper functions

def detect_faces(encodings, image=None):

    if image is not None:

        small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)

        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    else:

        video_capture = cv2.VideoCapture(0)

        while True:

            ret, frame = video_capture.read()

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)

            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

                matches = face_recognition.compare_faces(list(encodings.values()), face_encoding)

                name = "Unknown"

                if True in matches:

                    first_match_index = matches.index(True)

                    name = list(encodings.keys())[first_match_index]

                cv2.rectangle(frame, (left*4, top*4), (right*4, bottom*4), (0, 0, 255), 2)

                cv2.putText(frame, name, (left*4, bottom*4 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):

                break

        video_capture.release()

        cv2.destroyAllWindows()

        return None

    attendance = {}

    for face_encoding in face_encodings:

        matches = face_recognition.compare_faces(list(encodings.values()), face_encoding)

        name = "Unknown"

        if True in matches:

            first_match_index = matches.index(True)

            name = list(encodings.keys())[first_match_index]

        attendance[name] = True

    return attendance

def upload_image():

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        image = face_recognition.load_image_file(uploaded_file)

        return image

    else:

        return None

def display_attendance(attendance):

    st.write("Attendance:")

    for name, present in attendance.items():

        if present:

            st.write("- " + name + " (present)")

# Main app

st.title("Online Attendance System using Face Recognition")


option = st.radio(

    "Choose an option:",

    ("Upload an image", "Use webcam")

)

if option == "Upload an image":

    image = upload_image()

    if image is not None:

        attendance = detect_faces(encodings, image)

        display_attendance(attendance)

else:

    attendance = detect_faces(encodings)

    display_attendance(attendance)
