import cv2
import sys
import os
font = cv2.FONT_HERSHEY_SIMPLEX
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

def take_input_images_and_assign_label(name,images_to_capture):
    counter = 0
    os.mkdir("/Users/avyactjain/workspace/Facial Recognition/training_data/"+(name))
    video_capture = cv2.VideoCapture(0)
    working_path = ("/Users/avyactjain/workspace/Facial Recognition/training_data/"+(name))
    
    while (images_to_capture > 0):
        # Capture frame-by-frame
            ret, frame = video_capture.read()
        
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=10,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame,'SUCCESS! FACE DETECTED',(10,500), font, 1,(0,255,0),2,cv2.LINE_AA)
                face = frame[y:y+h,x:x+w]
        
            # Display the resulting frame
            cv2.imshow('Video', frame)
            
#             if cv2.waitKey(33) & 0xFF == ord('q'):
#                 break
            if cv2.waitKey(1) & 0xFF == ord('c'):
                
                counter = counter + 1
                resized_image = cv2.resize(face, (418, 418)) 
                cv2.imwrite(os.path.join(working_path ,name+str(counter)+'.png'), resized_image)
#                cv2.imwrite("/Users/avyactjain/workspace/Facial Recognition/training_data/"+name+str(counter)+'.png',face)
                print("Image", counter ,"captured Successfully")
                images_to_capture = images_to_capture -1
                
        #         break

# When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    
    
def face_lock():
    counter = 1
    os.mkdir("/Users/avyactjain/workspace/Facial Recognition/testing_data/")
    video_capture = cv2.VideoCapture(0)
    working_path = ("/Users/avyactjain/workspace/Facial Recognition/testing_data/"+str(counter))
    
    while True:
        # Capture frame-by-frame
            ret, frame = video_capture.read()
        
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=10,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame,'SUCCESS! FACE DETECTED',(10,500), font, 1,(0,255,0),2,cv2.LINE_AA)
                face = frame[y:y+h,x:x+w]
        
            # Display the resulting frame
            cv2.imshow('Video', frame)
            
#             if cv2.waitKey(33) & 0xFF == ord('q'):
#                 break
            if cv2.waitKey(1) & 0xFF == ord('l'):
               counter = counter + 1
               resized_image = cv2.resize(face, (418, 418)) 
               cv2.imwrite(os.path.join(working_path ,name+str(counter)+'.png'), resized_image)
#                cv2.imwrite("/Users/avyactjain/workspace/Facial Recognition/training_data/"+name+str(counter)+'.png',face)
               print("Face", counter ,"Lokced Successfully")
               
        #         break

# When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
            
        
    
    
