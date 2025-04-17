import os
import re
import shutil

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from git.repo import Repo

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Generate RSA key pair and specify the storage path
def generate_rsa_keypair(key_path):
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


# Function to encrypt files, specifying the public key path
def encrypt_file(file_name, public_key_path):
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
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    with open(file_name + ".encrypted", "wb") as encrypted_file:
        encrypted_file.write(iv + encrypted_aes_key + encrypted_data)


# Function to decrypt files, specifying the private key path
def decrypt_file(encrypted_file_name, private_key_path):
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
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
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
    with open(os.path.splitext(encrypted_file_name)[0], "wb") as decrypted_file:
        decrypted_file.write(decrypted_data)


def check_path():
    """Get path and check 'FlagScale' in path."""
    global path
    pattern_1 = r".*FlagScale.*"
    a = re.match(pattern_1, path)
    if a is None:
        raise FileNotFoundError("the FlagScale is not in your path")


def process_commit_id(patch_commit_id, base_commit_id=None):
    """Limit the length of the commit ID to 8."""
    if base_commit_id is not None:
        if len(base_commit_id) >= 8:
            base_commit_id = base_commit_id[:8]
        else:
            raise ValueError("base_commit_id is less longer than 8")
    if len(patch_commit_id) >= 8:
        patch_commit_id = patch_commit_id[:8]
    else:
        raise ValueError("patch_commit_id is less longer than 8")
    if base_commit_id is not None:
        return patch_commit_id, base_commit_id
    else:
        return patch_commit_id


def git_init(path=None):
    """Git init the repo from path."""
    if path:
        if not os.path.exists(path):
            cwd = os.getcwd()
            new_path = os.path.join(cwd, path)
            if not os.path.exists(new_path):
                raise FileNotFoundError(new_path)
    check_path()
    try:
        repo = Repo(path)
    except:
        raise FileNotFoundError(path)
    assert not repo.bare
    return repo


def crete_tmp_dir(dir_path=None, tmp_str=None):
    global path
    if dir_path is None:
        tmp_path = os.path.join(path, "../tmp_flagscale")
    else:
        if tmp_str is not None:
            tmp_path = os.path.join(dir_path, tmp_str.replace("tmp", "tmp_flagscale"))
        else:
            tmp_path = os.path.join(dir_path, "../tmp_flagscale")
    if not os.path.isdir(tmp_path):
        os.makedirs(tmp_path)
    else:
        shutil.rmtree(tmp_path)
        os.makedirs(tmp_path)
    return tmp_path


def check_branch_name(repo, branch_name):
    """Check if branch_name exists in the repository."""
    branch_list = repo.git.branch("--list")
    if branch_name in branch_list:
        return True
    else:
        return False


def get_now_branch_name(repo):
    """Get the now branch name when use this function"""
    branch_list = repo.git.branch("--list").split("\n")
    for branch_name in branch_list:
        if "*" in branch_name:
            branch_name = branch_name.split()[-1]
            return branch_name
    return "main"


def save_patch_to_tmp(patch_name, patch_str, key_path=None):
    """Save patch str to tmp patch file."""
    tmp_path = crete_tmp_dir()
    file_name = os.path.join(tmp_path, patch_name)
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(patch_str)
    if key_path is not None:
        os.makedirs(
            key_path, exist_ok=True
        )  # Create the directory if it does not exist
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
        encrypt_file(file_name, public_key_path)  # Encrypt the specified file
        # Delete the original patch file
        os.remove(file_name)
        file_name = file_name + ".encrypted"

    return file_name, tmp_path


def save_unpatch_to_tmp(tmp_path, base_commit_id_dir, patch_file, key_path=None):
    """Save patch file to tmp directory."""
    file_name = os.path.join(base_commit_id_dir, patch_file)
    if key_path is not None:
        private_key_path = os.path.join(key_path, "private_key.pem")
        decrypt_file(file_name + ".encrypted", private_key_path)  # Decrypt the file
    try:
        shutil.copy(file_name, tmp_path)
    except:
        raise ValueError("{} cannot cp".format(file_name))
    tmp_file_name = os.path.join(tmp_path, patch_file)
    return tmp_file_name
