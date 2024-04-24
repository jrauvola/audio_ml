import speech_recognition as sr
import pyttsx3

# Initialize the recognizer
r = sr.Recognizer()

def record_text():
    # Loop in case of errors
    while(1):
        try:
            # use the microphone as source for input
            with sr.Microphone() as source2:
                # prepare recognizer to receive input (ambient noise)
                r.adjust_for_ambient_noise(source2, duration=0.2)

                # listens for user's input
                audio2 = r.listen(source2)

                # using Google to recognize audio
                MyText = r.recognize_google(audio2)

                return MyText

        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

        except sr.UnknownValueError:
            print("Unknown error occurred")
    return

def output_text(text):
    f = open("output.txt", "a") # appending text to end of file
    f.write(text)
    f.write("\n")
    f.close()
    return

while(1):
    text = record_text()
    output_text(text)

    print("Wrote text")


# Open an external bash shell and input $ touch output.txt  $ tail -F output.txt
# Input in bash shell = python Downloads/speech_to_text.py