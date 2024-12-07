from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import os

mesaj = []
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def add_message(image):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_length=100)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    mesaj.append(generated_text)

def split_image_by_text_rows(image, output_dir):
    # Load the image
    #image = Image.open(image_path)

    # Get text with bounding boxes
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    # Extract bounding boxes for each line
    lines = []
    for i in range(len(data['level'])):
        if data['level'][i] == 4:  # '4' corresponds to text lines
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            lines.append((x, y, w, h))

    # Sort lines by their vertical position (top)
    lines = sorted(lines, key=lambda box: box[1])

    # Crop the image for each line and save
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, (x, y, w, h) in enumerate(lines):
        cropped_image = image.crop((x, y, x + w, y + h))
        output_path = os.path.join(output_dir, f"line_{idx + 1}.png")
        #cropped_image.save(output_path)
        add_message(cropped_image)
        print(mesaj)
        print(f"Cropped image: {output_path}")

def process_pdf(pdf_path, output_dir):
    # Convert PDF to images (one image per page)
    images = convert_from_path(pdf_path)

    for page_idx, image in enumerate(images):
        page_image_path = os.path.join(output_dir, f"page_{page_idx + 1}.png")
        #image.save(page_image_path)
        print(f"Processed page image: {page_image_path}")

        # Split image by text rows
        split_image_by_text_rows(image, output_dir)


# Example usage

pdf_file = "legi4.pdf"
output_directory = "output"
try:
    process_pdf(pdf_file, output_directory)
except Exception as e:
    pass
mean_length = 10
mesajfinal = ""
if len(mesajfinal) != 0:
    mean_length = sum(len(s) for s in mesaj) / len(mesaj)
for prop in mesaj:
    if len(mesajfinal) == 0:
        mesajfinal = prop
        continue
    if len(prop) < mean_length / 2:
        prop = f" <h1> {prop} </h1> "
    mesajfinal = f"{mesajfinal[:-1]}{prop}" if mesajfinal[-1] != "-" else f"{mesajfinal} {prop}"

with open("solutie.txt", "w") as file:
    file.write(mesajfinal)
#split_image_by_text_rows(input_image, output_directory)




# load image from the IAM dataset





# image_files = [f for f in os.listdir(output_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
# for file in image_files:
#     image = Image.open(f"output/{file}").convert("RGB")
#     pixel_values = processor(image, return_tensors="pt").pixel_values
#     generated_ids = model.generate(pixel_values, max_length=100)
#
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#
#     mesaj.append(generated_text)



