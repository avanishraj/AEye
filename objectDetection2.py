import cv2
import speech_recognition as sr
from gtts import gTTS
import os
import openai
openai.api_key = "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"

classNames = []
classFile = "coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects: 
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img, str(round(confidence*100,2)),(box[0] + 200,box[1] + 30), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    
    return img,objectInfo

def speak(text):
    language = 'en'
    speech = gTTS(text = text, lang = language, slow = False)
    speech.save("text.mp3")
    os.system("start text.mp3")

def process_text(text):
    prompt = "Q: " + text + "\nA:"
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        temperature=0.7,
        max_tokens=1024,
        n=1,
        stop=None,
        timeout=10,
    )
    return response.choices[0].text.strip()

if __name__ == "__main__":
    r = sr.Recognizer()
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    objects = []
    while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img,0.45,0.2, True, objects)
        cv2.imshow("Output",img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("o"):
            text = input("Please say the name of the object: ")
            # with sr.Microphone() as source:
            #     print("Please say the name of the object")
            #     audio = r.listen(source)
            #     try:
            #         text = r.recognize_google(audio)
            #         print("You said: " + text)
            #         objects.append(text)
            #     except sr.UnknownValueError:
            #         print("Google Speech Recognition could not understand audio")
            #     except sr.RequestError as e:
            #         print("Could not request results from Google Speech Recognition service; {0}".format(e))
        elif key == ord("s"):
            if len(objectInfo) != 0:
                objectNames = [obj[1] for obj in objectInfo]
                prompt = "What are " + ", "  + text + "?"
                response = process_text(prompt)
                speak(response)
            else:
                speak("I don't see any objects.")
cap.release()
cv2.destroyAllWindows()
