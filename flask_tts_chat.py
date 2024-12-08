from flask import Flask, request, jsonify, send_file
from transformers import pipeline
import speech_recognition as sr
from datasets import load_dataset
import soundfile as sf
import torch
from deep_translator import GoogleTranslator
import io

app = Flask(__name__)

userLanguage = "en"

# Initialize components
question_answerer = pipeline("question-answering", model="deepset/roberta-base-squad2")
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
recognizer = sr.Recognizer()


@app.route("/ask_question", methods=["POST"])
def ask_question():
    # Get the question from the request
    question = request.json["question"]
    print(question)

    if not question:
        return jsonify({"error": "No question provided"}), 400

    print(f"Received question: {question}")

    # Sample context (could be dynamic or from a database)
    context = '''A cell is the fundamental unit of life. There are hundreds of millions of distinct types of live cells can be found. A multicellular organism is made up of these cells, whereas a unicellular organism is made up of a single cell. Each cell is distinct, with its own set of functions and characteristics. Unicellular organisms, the most common type of organism, are made up of prokaryotic cells. Every bacteria has a prokaryotic cell with basic components. Eukaryotes, on the other hand, are a more evolved type of cell that makes up multicellular organisms, while just a few unicellular species have complex parts. They evolved from prokaryotic cells, as there are numerous similarities between prokaryotic and eukaryotic cells. Every cell inside our body is surrounded by a cell membrane (Plasma). The cell membrane divides the material outside the cell, known as extracellular material, from the stuff inside the cell, known as intracellular material. It protects a cell’s integrity and regulates the transport of materials into and out of the cell. For the necessary exchange, all materials inside a cell must have accessibility to the cell membrane (the cell’s boundary). The cell wall is the most exterior layer of a plant cell. It is formed of cellulose and serves as mechanical support for the cell. It covers the cell membrane and helps keep the pressure within the cell constant. The nucleus, which is created by a nuclear membrane encircling a fluid nucleoplasm, is the cell’s control center. Deoxyribonucleic acid (DNA), the cell’s genetic material, is undoubtly found in the chromatin threads present inside the nucleus. The nucleolus is a concentrated area of ribonucleic acid (RNA) in the nucleus where ribosomes are formed. The nucleus dictates how a cell will function as well as its basic. The centrosome is a component of an animal cell. A cell in an animal may have one or two centrosomes that aid in mitosis. Chloroplasts are plant cell components that are green in hue. They assist in the creation of food in the presence of sunlight through photosynthesis. The cytoplasm is the gel-like fluid that fills the inside of a particular cell. It serves as a medium for chemical reactions. It works as a platform for other organelles to function within the cell. The cytoplasm of a cell is where all of the functions for cell proliferation, growth, and replication take place. Materials migrate within the cytoplasm via diffusion, a physical mechanism that can only travel short distances. The endoplasmic reticulum refers to the tubular structures found surrounding the nucleus that assist both plant and animal cells. Endoplasmic reticulum is classified into two types: smooth reticulum without associated ribosomes and rough endoplasmic reticulum with attached ribosomes. The golgi apparatus or bodies are flat vesicular structures layered one on top of the other. They release and store hormones and enzymes that aid in cell transport. The mitochondrial membrane is composed of two layers, the inner of which is folded to generate cristae. It is the cell’s powerhouse, where ATP is produced by cellular respiration. Ribosomes are
the component of a cell that holds RNA and aids in protein synthesis. A vacuole is a big and plentiful vesicle found in plant cells. It holds fluids and aids in the storage of substances, construction materials, and water. A plant and animal cell are distinguished by their cell walls, central vacuoles, and chloroplasts. The smallest unit of life is, in fact, the most crucial for life’s sustenance! Cell structure theories have evolved significantly over time. Cells were once thought to be simple membrane sacs containing fluid with a few floating particles, according to early biologists. Cells are vastly more intricate than this, according to today’s biologists.
Cells in the body come in a variety of sizes, shapes, and types. It incorporates characteristics from all cell types. A cell is made up of three parts: the cell membrane, the nucleus, and the cytoplasm that lies between the two. Within the cytoplasm, there are complicated arrangements of fine fibres as well as hundreds or even thousands of tiny yet unique structures known as organelles. '''  # Truncated for brevity

    # Process the question and context
    res = question_answerer(question=question, context=context)
    answer = res['answer']

    # Translate the answer to the user's language if needed
    answer_user_language = GoogleTranslator(source='en', target=userLanguage).translate(answer)

    # Generate speech
    speech = synthesiser(answer_user_language,
                         forward_params={"speaker_embeddings": speaker_embedding, "language": userLanguage})

    # Save the speech as a WAV file in memory
    audio_io = io.BytesIO()
    sf.write(audio_io, speech["audio"], samplerate=speech["sampling_rate"], format="wav")
    audio_io.seek(0)

    return send_file(audio_io, mimetype="audio/wav", as_attachment=True, download_name="response.wav")


@app.route("/record_speech", methods=["POST"])
def record_speech():
    with sr.Microphone() as source:
        print("Listening for question...")
        audio = recognizer.listen(source, timeout=10)

        try:
            # Recognize the speech and extract the question
            question = recognizer.recognize_google(audio, language=userLanguage)
            print(f"Recognized question: {question}")

            return jsonify({"question": question})

        except sr.UnknownValueError:
            return jsonify({"error": "Could not understand audio"}), 400
        except sr.RequestError:
            return jsonify({"error": "Speech recognition request failed"}), 500


if __name__ == "__main__":
    app.run(debug=True)
