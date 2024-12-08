from flask import Flask, request, jsonify, send_file
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import os
import io

app = Flask(__name__)

# Initialize the processor and model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

mesaj = []  # This list will store the recognized text from images


def add_message(image):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_length=100)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    mesaj.append(generated_text)


def split_image_by_text_rows(image, output_dir):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    lines = []
    for i in range(len(data['level'])):
        if data['level'][i] == 4:  # '4' corresponds to text lines
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            lines.append((x, y, w, h))

    # Sort lines by vertical position (top)
    lines = sorted(lines, key=lambda box: box[1])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, (x, y, w, h) in enumerate(lines):
        cropped_image = image.crop((x, y, x + w, y + h))
        add_message(cropped_image)  # Process the cropped image

    return mesaj


def process_pdf(pdf_path, output_dir):
    images = convert_from_path(pdf_path)
    all_text = []

    for page_idx, image in enumerate(images):
        all_text += split_image_by_text_rows(image, output_dir)

    return all_text


@app.route("/process_pdf", methods=["POST"])
def process_pdf_endpoint():
    # Expect a file in the request
    pdf_file = request.files.get("pdf")
    if not pdf_file:
        return jsonify({"error": "No PDF file provided"}), 400

    output_directory = "output"
    pdf_path = os.path.join(output_directory, pdf_file.filename)
    pdf_file.save(pdf_path)

    try:
        # Process the PDF and return the result
        extracted_text = process_pdf(pdf_path, output_directory)
        return jsonify({"extracted_text": extracted_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/process_image", methods=["POST"])
def process_image_endpoint(img_path):
    print("merge?")
    # Expect an image file in the request
    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "No image file provided"}), 400

    image = Image.open(image_file)
    output_directory = "output"
    extracted_text = split_image_by_text_rows(image, output_directory)
    print("merge?")
    return jsonify({"extracted_text": extracted_text})


if __name__ == "__main__":
    app.run(debug=True)
