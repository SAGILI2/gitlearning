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

# Step 3: Convert to QR Code and Save
def save_as_qrcode(data: str, output_path: str):
    img = qrcode.make(data)
    img.save(output_path)

# Main Process
def main():
    file_path = 'path_to_file.txt'  # Replace with your file path
    output_qr_path = 'output_qr.png'  # Path where you want to save the QR code
    
    # Read and encode the data
    data = read_from_file(file_path)
    encoded_data = base64_encode(data)
    
    # Convert to QR and save
    save_as_qrcode(encoded_data, output_qr_path)
    print(f"QR code saved to: {output_qr_path}")

# Execute the main process
if __name__ == "__main__":
    main()
