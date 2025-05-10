import os

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def generate_rsa_keypair(key_path):
    """
    Generate RSA key pair and save them to the specified directory.
    Args:
        key_path (str): The path where the generated keys will be saved.
    Returns:
        tuple: A tuple containing the paths to the private and public keys respectively.
    """
    private_key_path = os.path.join(key_path, "private_key.pem")
    public_key_path = os.path.join(key_path, "public_key.pem")
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=4096, backend=default_backend()
    )
    public_key = private_key.public_key()
    with open(private_key_path, "wb") as private_file:
        private_file.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    with open(public_key_path, "wb") as public_file:
        public_file.write(
            public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
    return private_key_path, public_key_path


def encrypt_file(file_name, key_path):
    """
    Encrypt a file using AES-256-CBC and RSA-OAEP.
    Args:
        file_name (str): The name of the file to be encrypted.
        key_path (str): The path where the generated keys will be saved.
    Returns:
        encrypted_file_path: The path to the encrypted file.
    """
    # Create the directory if it does not exist
    os.makedirs(key_path, exist_ok=True)
    # Check if the key files already exist
    private_key_exists = os.path.exists(os.path.join(key_path, "private_key.pem"))
    public_key_exists = os.path.exists(os.path.join(key_path, "public_key.pem"))
    if not private_key_exists or not public_key_exists:
        # Generate RSA key pair if they do not exist
        private_key_path, public_key_path = generate_rsa_keypair(key_path)
        print(
            f"Generated RSA keys:\nPrivate key: {private_key_path}\nPublic key: {public_key_path}"
        )
    else:
        private_key_path = os.path.join(key_path, "private_key.pem")
        public_key_path = os.path.join(key_path, "public_key.pem")
        print("RSA keys already exist.")

    aes_key = os.urandom(32)
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    with open(file_name, "rb") as file:
        original_data = file.read()
    padding_length = 16 - len(original_data) % 16
    padded_data = original_data + bytes([padding_length] * padding_length)
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    with open(public_key_path, "rb") as public_file:
        public_key = serialization.load_pem_public_key(
            public_file.read(), backend=default_backend()
        )
    encrypted_aes_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None
        ),
    )
    encrypted_file_path = file_name + ".encrypted"
    with open(encrypted_file_path, "wb") as encrypted_file:
        encrypted_file.write(iv + encrypted_aes_key + encrypted_data)
        os.remove(file_name)
        print(f"Remove file {file_name} after file encrypted.")
    return encrypted_file_path


def decrypt_file(encrypted_file_name, private_key_path):
    """
    Decrypt a file using AES-256-CBC and RSA-OAEP.
    Args:
        encrypted_file_name (str): The name of the encrypted file.
        private_key_path (str): The path to the private key file.
    Returns:
        decrypted_file_path: The path to the decrypted file.
    """
    with open(private_key_path, "rb") as private_file:
        private_key = serialization.load_pem_private_key(
            private_file.read(), password=None, backend=default_backend()
        )
    with open(encrypted_file_name, "rb") as encrypted_file:
        print(encrypted_file_name, encrypted_file)
        iv = encrypted_file.read(16)
        encrypted_aes_key = encrypted_file.read(512)
        encrypted_data = encrypted_file.read()

    aes_key = None
    try:
        aes_key = private_key.decrypt(
            encrypted_aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None
            ),
        )
        print("AES key decrypted successfully.")
    except Exception as e:
        print(f"Decryption of AES key failed: {e}")
        return  # Exit the function if decryption fails
    if aes_key is None:
        print("AES key could not be decrypted.")
        return
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    try:
        decrypted_padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
    except Exception as e:
        print(f"Decryption of data failed: {e}")
        return
    padding_length = decrypted_padded_data[-1]
    decrypted_data = decrypted_padded_data[:-padding_length]
    decrypted_file_path = os.path.splitext(encrypted_file_name)[0]
    with open(decrypted_file_path, "wb") as decrypted_file:
        decrypted_file.write(decrypted_data)

    return decrypted_file_path
