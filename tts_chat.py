from transformers import pipeline
import speech_recognition as sr
from datasets import load_dataset
import soundfile as sf
import torch
import time #todo de scos
from deep_translator import GoogleTranslator

start = time.time()

userLanguage = "en"

question_answerer = pipeline("question-answering", model="deepset/roberta-base-squad2")
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
recognizer = sr.Recognizer()

print(f'setup time: {time.time() - start}')

# record 5 sec
question = ""
with sr.Microphone() as source:
    print("Say 'stop' to end the recording.")
    while True:
        try:
            print("Listening...")
            audio = recognizer.listen(source, timeout=10)  # Capture audio for 5 seconds

            # Try recognizing the speech
            try:
                question = recognizer.recognize_google(audio, language=userLanguage)
                print(f"You said: {question}")

                if "stop" in question.lower():  # Stop condition
                    print("Stopping recording...")
                    break

            except sr.UnknownValueError:
                print("Sorry, I didn't understand that.")
            except sr.RequestError:
                print("Could not request results; check your network connection.")
        except sr.WaitTimeoutError:
            print("Timeout reached, no speech detected.")


# question = GoogleTranslator(source=userLanguage, target='en').translate(question)
# question = question.replace("stop", "")
question = "what are the three parts of a cell stop"
print("Transcription: " + question)


context = '''A cell is the fundamental unit of life. There are hundreds of millions of distinct types of live cells can be found. A multicellular organism is made up of these cells, whereas a unicellular organism is made up of a single cell. Each cell is distinct, with its own set of functions and characteristics. Unicellular organisms, the most common type of organism, are made up of prokaryotic cells. Every bacteria has a prokaryotic cell with basic components. Eukaryotes, on the other hand, are a more evolved type of cell that makes up multicellular organisms, while just a few unicellular species have complex parts. They evolved from prokaryotic cells, as there are numerous similarities between prokaryotic and eukaryotic cells. Every cell inside our body is surrounded by a cell membrane (Plasma). The cell membrane divides the material outside the cell, known as extracellular material, from the stuff inside the cell, known as intracellular material. It protects a cell’s integrity and regulates the transport of materials into and out of the cell. For the necessary exchange, all materials inside a cell must have accessibility to the cell membrane (the cell’s boundary). The cell wall is the most exterior layer of a plant cell. It is formed of cellulose and serves as mechanical support for the cell. It covers the cell membrane and helps keep the pressure within the cell constant. The nucleus, which is created by a nuclear membrane encircling a fluid nucleoplasm, is the cell’s control center. Deoxyribonucleic acid (DNA), the cell’s genetic material, is undoubtly found in the chromatin threads present inside the nucleus. The nucleolus is a concentrated area of ribonucleic acid (RNA) in the nucleus where ribosomes are formed. The nucleus dictates how a cell will function as well as its basic. The centrosome is a component of an animal cell. A cell in an animal may have one or two centrosomes that aid in mitosis. Chloroplasts are plant cell components that are green in hue. They assist in the creation of food in the presence of sunlight through photosynthesis. The cytoplasm is the gel-like fluid that fills the inside of a particular cell. It serves as a medium for chemical reactions. It works as a platform for other organelles to function within the cell. The cytoplasm of a cell is where all of the functions for cell proliferation, growth, and replication take place. Materials migrate within the cytoplasm via diffusion, a physical mechanism that can only travel short distances. The endoplasmic reticulum refers to the tubular structures found surrounding the nucleus that assist both plant and animal cells. Endoplasmic reticulum is classified into two types: smooth reticulum without associated ribosomes and rough endoplasmic reticulum with attached ribosomes. The golgi apparatus or bodies are flat vesicular structures layered one on top of the other. They release and store hormones and enzymes that aid in cell transport. The mitochondrial membrane is composed of two layers, the inner of which is folded to generate cristae. It is the cell’s powerhouse, where ATP is produced by cellular respiration. Ribosomes are
the component of a cell that holds RNA and aids in protein synthesis. A vacuole is a big and plentiful vesicle found in plant cells. It holds fluids and aids in the storage of substances, construction materials, and water. A plant and animal cell are distinguished by their cell walls, central vacuoles, and chloroplasts. The smallest unit of life is, in fact, the most crucial for life’s sustenance! Cell structure theories have evolved significantly over time. Cells were once thought to be simple membrane sacs containing fluid with a few floating particles, according to early biologists. Cells are vastly more intricate than this, according to today’s biologists.
Cells in the body come in a variety of sizes, shapes, and types. It incorporates characteristics from all cell types. A cell is made up of three parts: the cell membrane, the nucleus, and the cytoplasm that lies between the two. Within the cytoplasm, there are complicated arrangements of fine fibres as well as hundreds or even thousands of tiny yet unique structures known as organelles. '''

contextEnglish = GoogleTranslator(source=userLanguage, target='en').translate(context)
res = question_answerer(question=question + '?', context=context)
resUserLanguage = res
resUserLanguage['answer'] = GoogleTranslator(source='en', target=userLanguage).translate(res['answer'])

print(resUserLanguage['answer'])

speech = synthesiser(resUserLanguage['answer'], forward_params={"speaker_embeddings": speaker_embedding, "language": userLanguage})
sf.write("response.wav", speech["audio"], samplerate=speech["sampling_rate"])
