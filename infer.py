from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import torch
import os

# Load the model and processor
def load_model():
    # Load the model with specific settings
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "./Qwen2.5-VL-7B-Instruct-abliterated",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Load the processor
    processor = AutoProcessor.from_pretrained(
        "./Qwen2.5-VL-7B-Instruct-abliterated",
        use_fast=True
    )

    # Return model and processor
    return model, processor

# Generate a descriptive paragraph for a given image
def generate_description(image_path, model, processor):
    # Construct the chat prompt with image and task instruction
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{os.path.abspath(image_path)}"
                },
                {
                    "type": "text",
                    "text": (
                        "Provide a clear description of this image in a dense paragraph. "
                        "Mention significant elements present, such as the environment, background, etc."
                    )
                }
            ]
        }
    ]

    # Format the text prompt
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Process visual inputs
    image_inputs, _ = process_vision_info(messages)

    # Tokenize and batch inputs
    inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=None,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    # Generate model output
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256
    )

    # Trim the input tokens from output
    trimmed_ids = [
        output[len(input_ids):]
        for input_ids, output in zip(inputs.input_ids, output_ids)
    ]

    # Decode the trimmed output
    description = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # Return the generated description
    return description

# Process all images in a directory and save their descriptions
def main():
    # Define the input image folder
    image_folder = "images"

    # Load the model and processor
    model, processor = load_model()

    # Define valid image file extensions
    valid_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")

    # List all valid image files
    image_files = [
        file for file in os.listdir(image_folder)
        if file.lower().endswith(valid_extensions)
    ]

    # Iterate through each image with a progress bar
    for image_file in tqdm(image_files, desc="Processing images"):
        # Build full path to the image
        image_path = os.path.join(image_folder, image_file)

        # Generate image description
        description = generate_description(image_path, model, processor)

        # Define path for output .txt file
        output_filename = os.path.splitext(image_file)[0] + ".txt"
        output_path = os.path.join(image_folder, output_filename)

        # Write the description to file
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(description)

        # Print the result to console
        print(f"\nImage: {image_file}\nDescription: {description}\n")

# Execute the main function
if __name__ == "__main__":
    main()
