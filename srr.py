import base64
import qrcode

# Function to encode data using Base64
def base64_encode(data: str) -> str:
    encoded_data = base64.b64encode(data.encode('utf-8'))
    return encoded_data.decode('utf-8')

# Step 1: Read Data from File
def read_from_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()

# Save encoded data to a file
def save_encoded_data(data: str, output_path: str):
    with open(output_path, 'w') as file:
        file.write(data)

# Step 3: Convert to QR Codes and Save
def save_as_qrcodes(data: str, base_output_path: str, chunk_size: int = 2000):
    # Splitting data into chunks
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    for index, chunk in enumerate(chunks):
        img = qrcode.make(chunk)
        img.save(f"{base_output_path}_{index}.png")
        print(f"QR code {index} saved to: {base_output_path}_{index}.png")

# Main Process
def main():
    file_path = 'path_to_file.txt'  # Replace with your file path
    output_qr_base_path = 'output_qr'  # Base path for QR codes (they will be named output_qr_0.png, output_qr_1.png, etc.)
    output_encoded_path = 'encoded_data.txt'  # Path to save the encoded data
    
    # Read and encode the data
    data = read_from_file(file_path)
    encoded_data = base64_encode(data)
    
    # Save encoded data to a file
    save_encoded_data(encoded_data, output_encoded_path)
    print(f"Encoded data saved to: {output_encoded_path}")
    
    # Convert to QR and save
    save_as_qrcodes(encoded_data, output_qr_base_path)

# Execute the main process
if __name__ == "__main__":
    main()
